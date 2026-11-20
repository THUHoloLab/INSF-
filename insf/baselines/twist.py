#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import torch

from insf.baselines.common import FRAMES33, NK, TUNE_FRAMES, Data, Writer, flat_init, metric_row, tv_prox, tv_value


def objective(a, x, y, wgt, lam, mu):
    r = (a * x).sum(0) - y
    return 0.5 * float((wgt * r * r).sum()) + lam * tv_value(x, mu)


def solve(a, y, wgt, lam, mu, iters, inner, xi1=1e-4):
    kappa = xi1
    rho = (1 - kappa ** 0.5) / (1 + kappa ** 0.5)
    alpha = rho * rho + 1
    beta = 2 * alpha / (xi1 + 1.0)
    x = flat_init(a, y, wgt)
    p = None
    grad_step = lambda z: z + a * (wgt * (y - (a * z).sum(0)))[None]
    x_prev = x
    x, p = tv_prox(grad_step(x), lam, mu, inner, p)
    f_old = objective(a, x, y, wgt, lam, mu)
    for _ in range(iters - 1):
        gam, p_new = tv_prox(grad_step(x), lam, mu, inner, p)
        x_new = (1 - alpha) * x_prev + (alpha - beta) * x + beta * gam
        f_new = objective(a, x_new, y, wgt, lam, mu)
        if f_new > f_old:
            x_new, f_new = gam, objective(a, gam, y, wgt, lam, mu)
        x_prev, x, p, f_old = x, x_new, p_new, f_new
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--inner", type=int, default=10)
    ap.add_argument("--lams", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    ap.add_argument("--mus", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    ap.add_argument("--frames", type=int, nargs="+", default=FRAMES33)
    ap.add_argument("--tune-frames", type=int, nargs="+", default=list(TUNE_FRAMES))
    ap.add_argument("--max-frames", type=int)
    args = ap.parse_args()
    run(args, solve, "twist", {"iters": args.iters, "inner": args.inner})


def run(args, solver, method, meta):
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    D = Data(dev, args.max_frames)
    args.out.mkdir(parents=True, exist_ok=True)
    tune_path = args.out / "tuning.json"
    kw = {k: v for k, v in meta.items()}

    if tune_path.exists():
        tune = json.loads(tune_path.read_text())
    else:
        tune = {"grid": [], "note": "score = mean over tune frames of RMSE (raw units) on 5% "
                "validation subset of TRAINING pixels; test pixels have zero weight throughout"}
    tried = {(g["lam"], g["mu"]) for g in tune["grid"]}
    for lam, mu in itertools.product(args.lams, args.mus):
        if (lam, mu) in tried:
            continue
        scores = []
        for t in args.tune_frames:
            v = D.val_mask(t)
            wgt = ((~D.hold[t]) & (~v)).float().reshape(D.h, D.w)
            ms = D.meas_scale(t)
            y = (D.frames[t] / ms).reshape(D.h, D.w) * wgt
            x = solver(D.a, y, wgt, lam, mu, **kw)
            cube = D.to_cube601(x, t)
            pred = (cube * D.M601).sum(-1)
            scores.append(float((pred - D.frames[t])[v].square().mean().sqrt()))
        tune["grid"].append({"lam": lam, "mu": mu, "val_rmse": float(sum(scores) / len(scores)),
                             "per_frame": scores})
        tune_path.write_text(json.dumps(tune, indent=2))
        print(f"[{method} tune] lam={lam:g} mu={mu:g} val_rmse={tune['grid'][-1]['val_rmse']:.4f}", flush=True)
    best = min(tune["grid"], key=lambda g: g["val_rmse"])
    tune["chosen"] = best
    tune_path.write_text(json.dumps(tune, indent=2))
    print(f"[{method}] chosen lam={best['lam']:g} mu={best['mu']:g}", flush=True)

    W = Writer(args.out, method, {"chosen": best, **meta, "parametrisation": f"{NK} knots 350:10:950 nm, linear to 601"})
    for t in args.frames:
        if t >= D.T or t in W.done():
            continue
        wgt = (~D.hold[t]).float().reshape(D.h, D.w)
        ms = D.meas_scale(t)
        y = (D.frames[t] / ms).reshape(D.h, D.w) * wgt
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x = solver(D.a, y, wgt, best["lam"], best["mu"], **kw)
        torch.cuda.synchronize()
        fit = time.perf_counter() - t0
        cube = D.to_cube601(x, t)
        row, pred = metric_row(D, t, cube, fit, meta.get("iters", 0), meta.get("iters", 0))
        W.add(D, t, cube, pred, row)
    W.summary()


if __name__ == "__main__":
    main()
