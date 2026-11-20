#!/usr/bin/env python3
from __future__ import annotations

import json

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
from insf.baselines.common import FRAMES33, NK, Data, Writer, metric_row

from insf import paths
EXT = paths.THIRD_PARTY
HARVARD = paths.HARVARD
C_IN = 2 + 2 * NK


def load_arch(name):
    import importlib.util
    path = {"mst": EXT / "MST/simulation/train_code/architecture/MST.py",
            "mstpp": EXT / "MST-plus-plus/train_code/architecture/MST_Plus_Plus.py"}[name]
    spec = importlib.util.spec_from_file_location(f"official_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.shift_back = lambda inputs, step=2: inputs
    return mod


class MSTNet(nn.Module):
    def __init__(self, dim=28, stage=3, num_blocks=(2, 4, 4)):
        super().__init__()
        mod = load_arch("mst")
        self.net = mod.MST(dim=dim, stage=stage, num_blocks=list(num_blocks))
        self.net.embedding = nn.Conv2d(C_IN, dim, 3, 1, 1, bias=False)
        self.net.mapping = nn.Conv2d(dim, NK, 3, 1, 1, bias=False)
        self.mask_proj = nn.Conv2d(NK, dim, 1, 1, 0, bias=True)

    def forward(self, inp, a, x0):
        n = self.net
        mask = self.mask_proj(a[:1])
        fea = n.lrelu(n.embedding(inp))
        fea_encoder, masks = [], []
        for (msab, fdown, mdown) in n.encoder_layers:
            fea = msab(fea, mask)
            masks.append(mask)
            fea_encoder.append(fea)
            fea = fdown(fea)
            mask = mdown(mask)
        fea = n.bottleneck(fea, mask)
        for i, (fup, fusion, blk) in enumerate(n.decoder_layers):
            fea = fup(fea)
            fea = fusion(torch.cat([fea, fea_encoder[n.stage - 1 - i]], dim=1))
            fea = blk(fea, masks[n.stage - 1 - i])
        return n.mapping(fea) + x0


class MSTPPNet(nn.Module):
    def __init__(self, stage=3):
        super().__init__()
        mod = load_arch("mstpp")
        net = mod.MST_Plus_Plus(in_channels=C_IN, out_channels=NK, n_feat=NK, stage=stage)
        net.body = nn.Sequential(*[mod.MST(in_dim=NK, out_dim=NK, dim=NK, stage=2, num_blocks=[1, 1, 1])
                                   for _ in range(stage)])
        self.net = net

    def forward(self, inp, a, x0):
        return self.net(inp)


def make_input(y, v, a):
    x0 = a * (y / a.square().sum(1).clamp_min(1e-8))[:, None]
    a = a.expand(y.shape[0], -1, -1, -1)
    return torch.cat([(y * v)[:, None], v[:, None], x0, a], 1), x0


def load_harvard(dev, max_scenes=None, exclude_holdout=False):
    from scipy.io import loadmat
    files = sorted(HARVARD.glob("img*.mat"))
    if exclude_holdout:
        hold = set(json.load(open(paths.HARVARD_HOLDOUT))["held_out"])
        files = [f for f in files if f.name not in hold]
    files = files[:max_scenes]
    cubes = []
    t0 = time.time()
    for f in files:
        d = loadmat(f, variable_names=["ref"])["ref"].astype(np.float32)
        d = d / max(float(np.percentile(d, 99.5)), 1e-6)
        cubes.append(torch.from_numpy(d).permute(2, 0, 1).to(dev, torch.float16))
    print(f"loaded {len(cubes)} Harvard cubes in {time.time()-t0:.0f}s", flush=True)
    return cubes


def to_knots(c31):
    lo = c31[:, :1].expand(-1, 7, -1, -1)
    hi = c31[:, -1:].expand(-1, NK - 7 - 31, -1, -1)
    return torch.cat([lo, c31, hi], 1)


def random_illum(B, dev, order=5, s=0.6):
    lam = torch.linspace(0, 1, NK, device=dev)
    basis = torch.stack([torch.cos(math.pi * j * lam) for j in range(1, order + 1)])
    coef = torch.randn(B, order, device=dev) * s / torch.arange(1, order + 1, device=dev)
    return torch.exp(coef @ basis)


def sample_batch(cubes, a_full, c, B, P, dev, gen):
    H, W = a_full.shape[-2:]
    r0 = int(torch.randint(0, H - P + 1, (1,), generator=gen))
    c0 = int(torch.randint(0, W - P + 1, (1,), generator=gen))
    a = a_full[None, :, r0:r0 + P, c0:c0 + P]
    xs = []
    for _ in range(B):
        cu = cubes[int(torch.randint(len(cubes), (1,), generator=gen))]
        sc = 2 if torch.rand((), generator=gen) < 0.5 else 1
        h, w = cu.shape[-2:]
        i = int(torch.randint(0, h - P * sc + 1, (1,), generator=gen))
        j = int(torch.randint(0, w - P * sc + 1, (1,), generator=gen))
        patch = cu[None, :, i:i + P * sc, j:j + P * sc].float()
        if sc == 2:
            patch = F.avg_pool2d(patch, 2)
        if torch.rand((), generator=gen) < 0.5:
            patch = patch.flip(-1)
        if torch.rand((), generator=gen) < 0.5:
            patch = patch.flip(-2)
        xs.append(patch)
    X = to_knots(torch.cat(xs)) * random_illum(B, dev)[:, :, None, None]
    y = c * (a * X).sum(1)
    ms = torch.quantile(y.abs().reshape(B, -1)[:, ::7], 0.99, dim=1).clamp_min(1e-6)
    y = y / ms[:, None, None]
    x = X * (c / ms)[:, None, None, None]
    k = 10 ** (torch.rand(B, 1, 1, device=dev) * 2 - 4.0)
    s0 = torch.rand(B, 1, 1, device=dev) * 0.01
    y = y + torch.randn_like(y) * (k * y.clamp_min(0) + s0 ** 2).sqrt()
    v = (torch.rand_like(y) >= 0.10).float()
    y = y * v
    return y, v, a, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mst", "mstpp"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-until", type=float, default=0, help="unix time to stop training")
    ap.add_argument("--max-steps", type=int, default=10 ** 9)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--ckpt-every", type=float, default=300, help="seconds")
    ap.add_argument("--infer-only", action="store_true")
    ap.add_argument("--max-scenes", type=int)
    ap.add_argument("--exclude-harvard-holdout", action="store_true")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--frames", type=int, nargs="+", default=FRAMES33)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    D = Data(dev, args.max_frames)
    net = (MSTNet() if args.arch == "mst" else MSTPPNet()).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    print(f"{args.arch}: {nparam/1e6:.2f} M parameters", flush=True)
    ck_path = args.out / "ckpt_last.pt"
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    step, t_used = 0, 0.0
    if ck_path.exists():
        ck = torch.load(ck_path, map_location=dev)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, t_used = ck["step"], ck["t_used"]
        print(f"resumed step {step}", flush=True)

    def save():
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "step": step,
                    "t_used": t_used}, str(ck_path) + ".tmp")
        Path(str(ck_path) + ".tmp").replace(ck_path)

    log = open(args.out / "train_log.txt", "a")
    if not args.infer_only:
        cubes = load_harvard(dev, args.max_scenes, args.exclude_harvard_holdout)
        gen = torch.Generator().manual_seed(2026 + step)
        t_start = time.time()
        budget = max(args.train_until - t_start, 1.0) + t_used if args.train_until else None
        last_ck, acc, n = time.time(), 0.0, 0
        net.train()
        while step < args.max_steps:
            now = time.time()
            if args.train_until and now >= args.train_until:
                break
            frac = (t_used + now - t_start) / budget if budget else step / args.max_steps
            lr = 1e-6 + 0.5 * (args.lr - 1e-6) * (1 + math.cos(math.pi * min(frac, 1.0)))
            for g in opt.param_groups:
                g["lr"] = lr
            y, v, a, x = sample_batch(cubes, D.a, D.c, args.batch, args.patch, dev, gen)
            inp, x0 = make_input(y, v, a)
            pred = net(inp, a, x0)
            loss = F.l1_loss(pred, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            step += 1
            acc += float(loss.detach()) if step % 50 == 0 else 0.0
            n += step % 50 == 0
            if step % 500 == 0:
                msg = f"step {step} loss {acc/max(n,1):.5f} lr {lr:.2e} t {t_used + time.time()-t_start:.0f}s"
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
                acc, n = 0.0, 0
            if time.time() - last_ck > args.ckpt_every:
                t_used_now = t_used + time.time() - t_start
                tu, t_used = t_used, t_used_now
                save()
                t_used = tu
                last_ck = time.time()
        t_used = t_used + time.time() - t_start
        save()
        del cubes
        torch.cuda.empty_cache()
        print(f"training done: {step} steps, {t_used:.0f}s", flush=True)

    net.eval()
    W = Writer(args.out, args.arch, {"steps": step, "train_seconds": t_used, "parameters": nparam,
                                      "train_data": "Harvard CZ_hsdb 420-720 nm, edge-held + random illuminant"})
    for t in args.frames:
        if t >= D.T:
            continue
        ms = D.meas_scale(t)
        v = (~D.hold[t]).float().reshape(1, D.h, D.w)
        y = (D.frames[t] / ms).reshape(1, D.h, D.w) * v
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            inp, x0 = make_input(y, v, D.a[None])
            x = net(inp, D.a[None], x0)[0]
        torch.cuda.synchronize()
        fit = time.perf_counter() - t0
        cube = D.to_cube601(x, t)
        row, pred = metric_row(D, t, cube, fit, 0, step)
        W.add(D, t, cube, pred, row)
    W.summary()


if __name__ == "__main__":
    main()
