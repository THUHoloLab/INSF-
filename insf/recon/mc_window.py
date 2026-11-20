#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from insf.recon.single_frame import (Config, build_2d, charbonnier, cosine_lr, frame_scales,
                       load_sequence, make_xy, render_cube, set_seed, sync_time, count_parameters)

OFFSET_W = {0: 1.0, 1: 0.7, 2: 0.5}
PREVIEW_NM = (450, 550, 650, 750, 850)


def spatial_tv(model, xy, wl, h, w, sig_scale, cfg):
    band = wl[torch.randint(wl.numel(), (1,), device=xy.device)]
    img = model(xy, band).reshape(h, w)
    return 0.5 * (charbonnier((img[:, 1:] - img[:, :-1]) / sig_scale, cfg.tv_epsilon)
                  + charbonnier((img[1:] - img[:-1]) / sig_scale, cfg.tv_epsilon))


class Window(torch.nn.Module):

    def __init__(self, offsets, grid):
        super().__init__()
        self.offsets = list(offsets)
        self.corr = torch.nn.ParameterDict(
            {str(d): torch.nn.Parameter(torch.zeros(1, 2, grid, grid)) for d in offsets})
        self.log_gain = torch.nn.ParameterDict({str(d): torch.nn.Parameter(torch.zeros(())) for d in offsets})
        self.bias = torch.nn.ParameterDict({str(d): torch.nn.Parameter(torch.zeros(())) for d in offsets})

    def correction(self, d, xy):
        g = xy[None, None] * 2 - 1
        return F.grid_sample(self.corr[str(d)], g, mode="bilinear", padding_mode="border",
                             align_corners=False)[0, :, 0].T

    def smooth(self):
        tot = 0.0
        for p in self.corr.values():
            tot = tot + (p[..., 1:, :] - p[..., :-1, :]).square().mean() + (p[..., 1:] - p[..., :-1]).square().mean()
        return tot / len(self.corr)

    def l2(self):
        return sum(p.square().mean() for p in self.corr.values()) / len(self.corr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["e1", "mc"], required=True)
    ap.add_argument("--frame-dir", type=Path, required=True)
    ap.add_argument("--mask", type=Path, required=True)
    ap.add_argument("--holdout", type=Path, required=True, help="bool [T,H,W] .npy, True = held out")
    ap.add_argument("--flow-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cube-dir", type=Path)
    ap.add_argument("--crop-size", type=int, default=512)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0, help="exclusive; 0 = all")
    ap.add_argument("--stride", type=int, default=1, help="e1 only: every n-th frame")
    ap.add_argument("--epochs", type=int, default=100, help="E1 epochs / first-window epochs")
    ap.add_argument("--warm-epochs", type=int, default=30)
    ap.add_argument("--warm-lr", type=float, default=3e-4)
    ap.add_argument("--radius", type=int, default=2, help="window half width (K = 2r+1)")
    ap.add_argument("--nb-frac", type=float, default=0.5, help="neighbour batch / centre batch")
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--flow-lr-mult", type=float, default=10.0)
    ap.add_argument("--flow-smooth", type=float, default=1e-3)
    ap.add_argument("--flow-l2", type=float, default=1e-5)
    ap.add_argument("--gain-prior", type=float, default=1e-3)
    ap.add_argument("--bias-prior", type=float, default=1e-4)
    ap.add_argument("--no-flow", action="store_true", help="window without warp")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: cap steps per window")
    args = ap.parse_args()

    cfg = Config()
    cfg.crop_size = args.crop_size
    if args.seed is not None:
        cfg.seed = args.seed
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    for sub in ("bands", "pred_measurement"):
        (args.out / sub).mkdir(exist_ok=True)
    if args.cube_dir:
        args.cube_dir.mkdir(parents=True, exist_ok=True)

    mask_np, frames_np, sat_np, paths, _ = load_sequence(args, cfg)
    T = len(paths)
    h, w, L = mask_np.shape
    n_pix = h * w
    mask_sum = mask_np.sum(-1)
    xy = make_xy(h, w, dev)
    wl = torch.linspace(0.0, 1.0, L, device=dev)
    mask_t = torch.from_numpy(mask_np.reshape(-1, L)).to(dev)
    frames_t = torch.from_numpy(frames_np.reshape(T, -1)).to(dev)
    sat_t = torch.from_numpy(sat_np.reshape(T, -1)).to(dev)
    hold = torch.from_numpy(np.load(args.holdout)[:T].reshape(T, -1)).to(dev)
    train_idx = [torch.nonzero(~hold[t]).squeeze(1) for t in range(T)]
    band_idx = [nm - 350 for nm in PREVIEW_NM]

    offsets = [d for d in range(-args.radius, args.radius + 1) if d != 0] if args.method == "mc" else []
    flows, confs = {}, {}
    if offsets:
        for d in offsets:
            fl = torch.from_numpy(np.load(args.flow_dir / f"flow_d{d:+d}.npy")[:T].astype(np.float32))
            if args.no_flow:
                fl.zero_()
            flows[d] = fl.reshape(T, -1, 2).to(dev)
            cf = torch.from_numpy(np.load(args.flow_dir / f"conf_d{d:+d}.npy")[:T])
            if args.no_flow:
                cf[:] = True
            confs[d] = cf.reshape(T, -1).to(dev)
    pix_scale = torch.tensor([1.0 / w, 1.0 / h], device=dev)

    log_f = open(args.out / "train_log.txt", "a")

    def log(msg, echo=True):
        log_f.write(msg + "\n")
        log_f.flush()
        if echo:
            print(msg, flush=True)

    (args.out / "run_meta.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, "config": cfg.to_dict(),
         "frames": [p.name for p in paths], "torch": torch.__version__}, indent=2, ensure_ascii=False))

    end = args.end or T
    rows = []
    model = opt = win = None
    done = set()
    ckpt_path = args.out / "ckpt_last.pt"
    csv_path = args.out / "metrics_per_frame.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            rows = [{k: (v if k == "file" else float(v)) if v != "" else float("nan")
                     for k, v in r.items()} for r in csv.DictReader(f)]
        for r in rows:
            r["frame"] = int(r["frame"])
        done = {r["frame"] for r in rows}
        if args.method == "mc" and rows:
            ck = torch.load(ckpt_path, map_location=dev)
            assert ck["frame"] == max(done), (ck["frame"], max(done))
            _, sig0 = frame_scales(frames_np[args.start], mask_sum)
            set_seed(cfg.seed)
            model = build_2d(cfg, sig0, dev)
            win = Window(offsets, args.grid).to(dev)
            groups = [{"params": list(model.parameters()), "mult": 1.0},
                      {"params": list(win.corr.parameters()), "mult": args.flow_lr_mult},
                      {"params": list(win.log_gain.parameters()) + list(win.bias.parameters()), "mult": 1.0}]
            opt = torch.optim.Adam(groups, lr=cfg.learning_rate)
            model.load_state_dict(ck["model"])
            win.load_state_dict(ck["win"])
            opt.load_state_dict(ck["opt"])
        log(f"resuming: {len(done)} frames already done")
    t_run = sync_time()
    for t in range(args.start, end, args.stride if args.method == "e1" else 1):
        if t in done:
            continue
        y_t = frames_t[t]
        meas_scale, sig_scale = frame_scales(frames_np[t], mask_sum)
        t0 = sync_time()
        fresh = args.method == "e1" or model is None
        if fresh:
            set_seed(cfg.seed)
            model = build_2d(cfg, sig_scale, dev)
            groups = [{"params": list(model.parameters()), "mult": 1.0}]
            if offsets:
                win = Window(offsets, args.grid).to(dev)
                groups.append({"params": list(win.corr.parameters()), "mult": args.flow_lr_mult})
                groups.append({"params": list(win.log_gain.parameters()) + list(win.bias.parameters()), "mult": 1.0})
            opt = torch.optim.Adam(groups, lr=cfg.learning_rate)
            epochs, lr0 = args.epochs, cfg.learning_rate
        else:
            epochs, lr0 = args.warm_epochs, args.warm_lr
        nbrs = [d for d in offsets if 0 <= t + d < T]
        idx_c = train_idx[t]
        steps = idx_c.numel() // cfg.batch_size
        total = epochs * steps
        if args.max_steps:
            total = min(total, args.max_steps)
        Bn = int(cfg.batch_size * args.nb_frac)
        wsum = 1.0 + sum(OFFSET_W[abs(d)] for d in nbrs)
        acc = torch.zeros(4, device=dev)
        perm = None
        for k in range(total):
            if k % steps == 0:
                perm = idx_c[torch.randperm(idx_c.numel(), device=dev)]
            lr = cosine_lr(k, total, lr0, cfg.min_learning_rate)
            for g in opt.param_groups:
                g["lr"] = lr * g["mult"]
            s0 = (k % steps) * cfg.batch_size
            p_c = perm[s0:s0 + cfg.batch_size]
            coords = [xy[p_c]]
            nb_info = []
            for d in nbrs:
                s = t + d
                tr = train_idx[s]
                p_s = tr[torch.randint(tr.numel(), (Bn,), device=dev)]
                u = flows[d][t][p_s]
                if not args.no_flow:
                    u = u + win.correction(d, xy[p_s])
                c = xy[p_s] + u * pix_scale
                inside = ((c >= 0) & (c <= 1)).all(1)
                coords.append(c.clamp(0, 1))
                nb_info.append((d, s, p_s, (confs[d][t][p_s] & inside).float()))
            opt.zero_grad(set_to_none=True)
            spec_all = model(torch.cat(coords), wl)
            spec = spec_all[:cfg.batch_size]
            est = (spec * mask_t[p_c]).sum(-1)
            data = F.smooth_l1_loss(est / meas_scale, y_t[p_c] / meas_scale, beta=cfg.huber_beta)
            data_tot = data
            reg = 0.0
            for j, (d, s, p_s, c) in enumerate(nb_info):
                sp = spec_all[cfg.batch_size + j * Bn: cfg.batch_size + (j + 1) * Bn]
                pred = win.log_gain[str(d)].exp() * (sp * mask_t[p_s]).sum(-1) + win.bias[str(d)]
                r = F.smooth_l1_loss(pred / meas_scale, frames_t[s, p_s] / meas_scale,
                                     beta=cfg.huber_beta, reduction="none")
                data_tot = data_tot + OFFSET_W[abs(d)] * (r * c).sum() / c.sum().clamp_min(1.0)
                reg = reg + args.gain_prior * win.log_gain[str(d)].square() \
                    + args.bias_prior * (win.bias[str(d)] / meas_scale).square()
            data_tot = data_tot / wsum
            stv = charbonnier(torch.diff(spec, dim=-1) / sig_scale, cfg.tv_epsilon)
            ptv = spatial_tv(model, xy, wl, h, w, sig_scale, cfg)
            loss = data_tot + cfg.spectral_tv_weight * stv + cfg.spatial_tv_weight * ptv + reg
            if nbrs and not args.no_flow:
                loss = loss + args.flow_smooth * win.smooth() + args.flow_l2 * win.l2()
            loss.backward()
            opt.step()
            acc += torch.stack([data.detach(), data_tot.detach(), stv.detach(), ptv.detach()])
            if (k + 1) % steps == 0:
                a = (acc / steps).tolist()
                log(f"  t={t} epoch {(k+1)//steps}/{epochs} lr={lr:.2e} data_c={a[0]:.5g} "
                    f"data_win={a[1]:.5g} spec_tv={a[2]:.5g} spat_tv={a[3]:.5g}", echo=False)
                acc.zero_()
        fit_s = sync_time() - t0

        cube = render_cube(model, xy, wl)
        pred = (cube * mask_t).sum(-1)
        res = pred - y_t
        ho = hold[t]
        unsat = ~sat_t[t]
        tr = ~ho
        row = {"frame": t, "file": paths[t].name, "epochs": epochs, "steps": total,
               "fit_seconds": fit_s, "n_neighbours": len(nbrs),
               "rmse_holdout": float(res[ho].square().mean().sqrt()),
               "rmse_holdout_unsat": float(res[ho & unsat].square().mean().sqrt()),
               "rmse_train": float(res[tr].square().mean().sqrt()),
               "rmse_all": float(res.square().mean().sqrt()),
               "rel_rmse_holdout": float(res[ho].square().mean().sqrt() / y_t.std()),
               "cube_mean": float(cube.mean())}
        for nm, bi in zip(PREVIEW_NM, band_idx):
            row[f"band_mean_{nm}"] = float(cube[:, bi].mean())
        if win is not None:
            for d in nbrs:
                row[f"corr_rms_d{d:+d}"] = float(win.corr[str(d)].square().mean().sqrt())
                row[f"gain_d{d:+d}"] = float(win.log_gain[str(d)].exp())
                row[f"conf_d{d:+d}"] = float(confs[d][t].float().mean())
        rows.append(row)
        np.save(args.out / "bands" / f"frame_{t:03d}.npy",
                cube[:, band_idx].reshape(h, w, -1).cpu().numpy().astype(np.float32))
        np.save(args.out / "pred_measurement" / f"frame_{t:03d}.npy",
                pred.reshape(h, w).cpu().numpy().astype(np.float32))
        if win is not None:
            (args.out / "window_state").mkdir(exist_ok=True)
            torch.save({k: v.detach().cpu() for k, v in win.state_dict().items()},
                       args.out / "window_state" / f"frame_{t:03d}.pt")
        if args.cube_dir:
            np.save(args.cube_dir / f"frame_{t:03d}.npy",
                    cube.reshape(h, w, -1).cpu().numpy().astype(np.float16))
        log(f"[frame {t:03d}] holdout={row['rmse_holdout']:.3f} train={row['rmse_train']:.3f} "
            f"nb={len(nbrs)} epochs={epochs} fit={fit_s:.1f}s")
        if args.method == "mc":
            torch.save({"frame": t, "model": model.state_dict(), "win": win.state_dict(),
                        "opt": opt.state_dict()}, str(ckpt_path) + ".tmp")
            Path(str(ckpt_path) + ".tmp").replace(ckpt_path)
        with open(csv_path, "w", newline="") as f:
            keys = []
            for r in rows:
                keys += [k for k in r if k not in keys]
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
    total_s = sync_time() - t_run
    summ = {"method": args.method, "frames": [args.start, end], "total_seconds": total_s,
            "mean_rmse_holdout": float(np.mean([r["rmse_holdout"] for r in rows])),
            "mean_rmse_train": float(np.mean([r["rmse_train"] for r in rows])),
            "parameters": count_parameters(model)}
    (args.out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
