from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

import os
from insf import paths
from insf.recon.single_frame import Config, frame_scales, load_sequence
R8 = paths.OUT / "mc_window"

FRAME_DIR = Path(os.environ.get("QS_FRAME_DIR", paths.MOVE2))
MASK = paths.MASK
HOLDOUT = Path(os.environ.get("QS_HOLDOUT", paths.HOLDOUT_MOVE2))
FRAMES33 = list(range(0, 98, 3))
FIG_FRAMES = (0, 24, 48, 72, 96)
TUNE_FRAMES = (0, 48, 96)
PREVIEW_NM = (450, 550, 650, 750, 850)
KNOTS_NM = np.arange(350, 951, 10)
BANDS_NM = np.arange(350, 951, 1)
NK = len(KNOTS_NM)


def interp_matrix() -> np.ndarray:
    B = np.zeros((NK, len(BANDS_NM)), np.float32)
    for l, nm in enumerate(BANDS_NM):
        f = (nm - KNOTS_NM[0]) / 10.0
        k = min(int(np.floor(f)), NK - 2)
        r = f - k
        B[k, l] += 1 - r
        B[k + 1, l] += r
    return B


class Data:

    def __init__(self, dev, max_frames=None):
        cfg = Config()
        cfg.crop_size = int(os.environ.get("QS_CROP", "512"))
        args = SimpleNamespace(mask=MASK, frame_dir=FRAME_DIR, max_frames=max_frames)
        mask, frames, sat, paths, _ = load_sequence(args, cfg)
        self.paths = paths
        self.T = len(paths)
        self.h, self.w, self.L = mask.shape
        assert self.L == 601
        self.mask_sum = mask.sum(-1)
        self.frames_np = frames
        self.dev = dev
        self.B = torch.from_numpy(interp_matrix()).to(dev)
        self.M601 = torch.from_numpy(mask.reshape(-1, self.L)).to(dev)
        self.M61 = self.M601 @ self.B.T
        self.c = float(self.M61.norm(dim=1).max())
        self.a = (self.M61 / self.c).T.reshape(NK, self.h, self.w).contiguous()
        self.frames = torch.from_numpy(frames.reshape(self.T, -1)).to(dev)
        self.sat = torch.from_numpy(sat.reshape(self.T, -1)).to(dev)
        self.hold = torch.from_numpy(np.load(HOLDOUT)[: self.T].reshape(self.T, -1)).to(dev)

    def meas_scale(self, t):
        return frame_scales(self.frames_np[t], self.mask_sum)[0]

    def val_mask(self, t, frac=0.05, seed=1234):
        g = torch.Generator(device="cpu").manual_seed(seed + t)
        tr = torch.nonzero(~self.hold[t].cpu()).squeeze(1)
        sel = tr[torch.randperm(tr.numel(), generator=g)[: int(frac * tr.numel())]]
        v = torch.zeros(self.h * self.w, dtype=torch.bool)
        v[sel] = True
        return v.to(self.dev)

    def to_cube601(self, x, t):
        X = x.reshape(NK, -1).T * (self.meas_scale(t) / self.c)
        return X @ self.B


def flat_init(a, y, wgt, k=7):
    s = a.sum(0).clamp_min(1e-6)
    f = torch.where(wgt > 0, y / s, torch.zeros_like(y))
    num = F.avg_pool2d((f * wgt)[None, None], k, 1, k // 2)[0, 0]
    den = F.avg_pool2d(wgt[None, None].float(), k, 1, k // 2)[0, 0].clamp_min(1e-6)
    f = torch.where(wgt > 0, f, num / den)
    return f[None].expand_as(a).clone()


def metric_row(D: Data, t, cube, fit_s, epochs, steps):
    y_t = D.frames[t]
    pred = (cube * D.M601).sum(-1)
    res = pred - y_t
    ho = D.hold[t]
    unsat = ~D.sat[t]
    tr = ~ho
    row = {"frame": t, "file": D.paths[t].name, "epochs": epochs, "steps": steps,
           "fit_seconds": fit_s, "n_neighbours": 0,
           "rmse_holdout": float(res[ho].square().mean().sqrt()),
           "rmse_holdout_unsat": float(res[ho & unsat].square().mean().sqrt()),
           "rmse_train": float(res[tr].square().mean().sqrt()),
           "rmse_all": float(res.square().mean().sqrt()),
           "rel_rmse_holdout": float(res[ho].square().mean().sqrt() / y_t.std()),
           "cube_mean": float(cube.mean())}
    for nm in PREVIEW_NM:
        row[f"band_mean_{nm}"] = float(cube[:, nm - 350].mean())
    return row, pred


class Writer:

    def __init__(self, out: Path, method: str, extra_meta=None):
        self.out, self.method = out, method
        out.mkdir(parents=True, exist_ok=True)
        for sub in ("bands", "band750", "pred_measurement"):
            (out / sub).mkdir(exist_ok=True)
        self.csv = out / "metrics_per_frame.csv"
        self.rows = []
        if self.csv.exists():
            with open(self.csv) as f:
                self.rows = [{k: (v if k == "file" else float(v)) for k, v in r.items()}
                             for r in csv.DictReader(f)]
            for r in self.rows:
                r["frame"] = int(r["frame"])
        self.meta = extra_meta or {}

    def done(self):
        return {r["frame"] for r in self.rows}

    def add(self, D: Data, t, cube, pred, row):
        h, w = D.h, D.w
        self.rows = [r for r in self.rows if r["frame"] != t] + [row]
        self.rows.sort(key=lambda r: r["frame"])
        np.save(self.out / "bands" / f"frame_{t:03d}.npy",
                cube[:, [nm - 350 for nm in PREVIEW_NM]].reshape(h, w, -1).cpu().numpy().astype(np.float32))
        np.save(self.out / "band750" / f"frame_{t:03d}.npy",
                cube[:, 400].reshape(h, w).cpu().numpy().astype(np.float32))
        np.save(self.out / "pred_measurement" / f"frame_{t:03d}.npy",
                pred.reshape(h, w).cpu().numpy().astype(np.float32))
        keys = []
        for r in self.rows:
            keys += [k for k in r if k not in keys]
        tmp = self.csv.with_suffix(".tmp")
        with open(tmp, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(self.rows)
        tmp.replace(self.csv)
        self.summary()
        print(f"[{self.method} frame {t:03d}] holdout={row['rmse_holdout']:.3f} "
              f"train={row['rmse_train']:.3f} fit={row['fit_seconds']:.1f}s", flush=True)

    def summary(self):
        def agg(sel):
            rs = [r for r in self.rows if sel(r["frame"])]
            if not rs:
                return {"n_frames": 0}
            d = {k: float(np.mean([r[k] for r in rs])) for k in
                 ("rmse_holdout", "rmse_holdout_unsat", "rmse_train", "rmse_all", "fit_seconds")}
            d["n_frames"] = len(rs)
            return d
        ref = {}
        try:
            s8 = json.loads((R8 / "postproc" / "summary_mc_vs_e1.json").read_text())
            ref = {k: s8[k] for k in ("shared33", "shared_fast40_85")}
        except Exception as e:
            ref = {"error": str(e)}
        summ = {"method": self.method,
                "shared33": agg(lambda t: t in FRAMES33),
                "shared_fast40_85": agg(lambda t: t in FRAMES33 and 40 <= t <= 85),
                "frames_done": [r["frame"] for r in self.rows],
                "complete": all(t in self.done() for t in FRAMES33),
                "meta": self.meta, "reference_08": ref}
        (self.out / "summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))


def grad(x, mu):
    dx = F.pad(x[:, :, 1:] - x[:, :, :-1], (0, 1))
    dy = F.pad(x[:, 1:] - x[:, :-1], (0, 0, 0, 1))
    ds = F.pad(x[1:] - x[:-1], (0, 0, 0, 0, 0, 1)) * mu
    return dx, dy, ds


def div(px, py, ps, mu):
    dx = torch.cat([px[:, :, :1], px[:, :, 1:-1] - px[:, :, :-2], -px[:, :, -2:-1]], 2)
    dy = torch.cat([py[:, :1], py[:, 1:-1] - py[:, :-2], -py[:, -2:-1]], 1)
    ds = torch.cat([ps[:1], ps[1:-1] - ps[:-2], -ps[-2:-1]], 0) * mu
    return dx + dy + ds


def tv_value(x, mu):
    dx, dy, ds = grad(x, mu)
    return float((dx.square() + dy.square()).sqrt().sum() + ds.abs().sum())


def tv_prox(f, lam, mu, iters=20, p=None):
    if lam <= 0:
        return f, p
    L = 8.0 + 4.0 * mu * mu
    if p is None:
        p = [torch.zeros_like(f) for _ in range(3)]
    r = [q.clone() for q in p]
    tk = 1.0
    for _ in range(iters):
        u = f + lam * div(*r, mu)
        g = grad(u, mu)
        q = [r[i] + g[i] / (lam * L) for i in range(3)]
        nrm = (q[0].square() + q[1].square()).sqrt().clamp_min(1.0)
        q = [q[0] / nrm, q[1] / nrm, q[2].clamp(-1, 1)]
        tk1 = (1 + (1 + 4 * tk * tk) ** 0.5) / 2
        r = [q[i] + (tk - 1) / tk1 * (q[i] - p[i]) for i in range(3)]
        p, tk = q, tk1
    return f + lam * div(*p, mu), p
