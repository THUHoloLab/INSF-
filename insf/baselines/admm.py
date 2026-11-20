#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from insf.baselines.common import FRAMES33, TUNE_FRAMES, div, flat_init, grad
from insf.baselines.twist import run


def solve(a, y, wgt, lam, mu, iters, cg_iters, rho_mult=10.0):
    rho = rho_mult * lam
    x = flat_init(a, y, wgt)
    z = list(grad(x, mu))
    u = [torch.zeros_like(q) for q in z]
    Aty = a * (wgt * y)[None]

    def op(v):
        return a * (wgt * (a * v).sum(0))[None] - rho * div(*grad(v, mu), mu)

    for _ in range(iters):
        rhs = Aty - rho * div(*[z[i] - u[i] for i in range(3)], mu)
        r = rhs - op(x)
        p = r.clone()
        rs = (r * r).sum()
        for _ in range(cg_iters):
            Ap = op(p)
            al = rs / (p * Ap).sum().clamp_min(1e-30)
            x = x + al * p
            r = r - al * Ap
            rs_new = (r * r).sum()
            p = r + (rs_new / rs.clamp_min(1e-30)) * p
            rs = rs_new
        g = grad(x, mu)
        v = [g[i] + u[i] for i in range(3)]
        nrm = (v[0].square() + v[1].square()).sqrt().clamp_min(1e-12)
        s = (1 - (lam / rho) / nrm).clamp_min(0)
        z = [v[0] * s, v[1] * s, torch.sign(v[2]) * (v[2].abs() - lam / rho).clamp_min(0)]
        u = [u[i] + g[i] - z[i] for i in range(3)]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--cg-iters", type=int, default=8)
    ap.add_argument("--lams", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    ap.add_argument("--mus", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    ap.add_argument("--frames", type=int, nargs="+", default=FRAMES33)
    ap.add_argument("--tune-frames", type=int, nargs="+", default=list(TUNE_FRAMES))
    ap.add_argument("--max-frames", type=int)
    args = ap.parse_args()
    run(args, solve, "admm", {"iters": args.iters, "cg_iters": args.cg_iters})


if __name__ == "__main__":
    main()
