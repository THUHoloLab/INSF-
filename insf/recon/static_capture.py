#!/usr/bin/env python3
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

from insf import paths
B09 = paths.OUT / "baselines"
from insf.baselines.common import NK, interp_matrix, frame_scales
from insf.baselines import twist, admm
from insf.baselines.mst import MSTNet, make_input
from insf.recon.single_frame import (Config, build_2d, fit_frame, render_cube, make_xy, set_seed, sync_time,
                       cie_matrix, SATURATION)
from insf.data.io import _load_mask, _load_measurement, centered_crop_slices

ap = argparse.ArgumentParser()
ap.add_argument("--meas", type=Path, default=paths.STATIC / "2025-08-15--10-55-46-597.mat")
ap.add_argument("--mask", type=Path, default=paths.MASK)
ap.add_argument("--xyzbar", type=Path, default=paths.XYZBAR)
ap.add_argument("--crop", type=int, default=512)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--meas-npy", type=Path, help="simulated measurement [crop,crop] (already x gain) instead of --meas")
ap.add_argument("--methods", nargs="+", default=["twist", "admm", "mst", "inr"])
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

cfg = Config(); cfg.crop_size = args.crop
rows, cols, r0, c0 = centered_crop_slices(cfg)
mask = _load_mask(args.mask, "dataCube", rows, cols)
raw = (np.load(args.meas_npy).astype(np.float32) / np.float32(cfg.measurement_gain)) if args.meas_npy else _load_measurement(args.meas, "measurement", rows, cols).astype(np.float32)
y_np = raw * np.float32(cfg.measurement_gain)
sat = raw >= SATURATION
h, w, L = mask.shape
mask_sum = mask.sum(-1)
ms, sig = frame_scales(y_np, mask_sum)
np.save(args.out / "measurement.npy", y_np)
M601 = torch.from_numpy(mask.reshape(-1, L)).to(dev)
Bm = torch.from_numpy(interp_matrix()).to(dev)
M61 = M601 @ Bm.T
c = float(M61.norm(dim=1).max())
a = (M61 / c).T.reshape(NK, h, w).contiguous()
y = torch.from_numpy(y_np.reshape(-1)).to(dev)
yn = (y / ms).reshape(h, w)
ones = torch.ones(h, w, device=dev)
rgbm = torch.from_numpy(cie_matrix(args.xyzbar)).to(dev)
unsat = torch.from_numpy(~sat.reshape(-1)).to(dev)
tw = json.loads((B09 / "runs/twist/tuning.json").read_text())["chosen"]
ad = json.loads((B09 / "runs/admm/tuning.json").read_text())["chosen"]
meta = {"measurement": str(args.meas_npy or args.meas), "crop_size": args.crop,
        "crop_rows_1based": [r0 + 1, r0 + h], "crop_cols_1based": [c0 + 1, c0 + w],
        "measurement_gain": cfg.measurement_gain, "meas_scale_p99": ms, "saturated_fraction": float(sat.mean()),
        "gpu": torch.cuda.get_device_name(0), "methods": {}}
mp = args.out / "meta.json"
if mp.exists():
    old = json.loads(mp.read_text()); meta["methods"] = old.get("methods", {})


def knots_to_cube(x):
    return (x.reshape(NK, -1).T * (ms / c)) @ Bm


def finish(name, cube, secs, settings):
    pred = (cube * M601).sum(-1)
    res = pred - y
    rgb = (cube @ rgbm).reshape(h, w, 3)
    np.save(args.out / f"{name}_rgb_linear.npy", rgb.cpu().numpy().astype(np.float32))
    np.save(args.out / f"{name}_cube.npy", cube.reshape(h, w, L).cpu().numpy().astype(np.float16))
    meta["methods"][name] = {"seconds": secs, "settings": settings,
                             "rmse_measurement_all": float(res.square().mean().sqrt()),
                             "rmse_measurement_unsat": float(res[unsat].square().mean().sqrt()),
                             "band_mean": {str(nm): float(cube[:, nm - 350].mean()) for nm in (450, 550, 650, 750, 850)}}
    mp.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(name, json.dumps(meta["methods"][name]), flush=True)


for m in args.methods:
    t0 = sync_time()
    if m == "twist":
        x = twist.solve(a, yn, ones, tw["lam"], tw["mu"], iters=400, inner=10)
        secs = sync_time() - t0
        finish(m, knots_to_cube(x), secs, {"lam": tw["lam"], "mu": tw["mu"], "iters": 400, "inner": 10})
    elif m == "admm":
        x = admm.solve(a, yn, ones, ad["lam"], ad["mu"], iters=150, cg_iters=8)
        secs = sync_time() - t0
        finish(m, knots_to_cube(x), secs, {"lam": ad["lam"], "mu": ad["mu"], "iters": 150, "cg_iters": 8, "rho_mult": 10})
    elif m == "mst":
        net = MSTNet().to(dev)
        ck = torch.load(B09 / "runs/mst/ckpt_last.pt", map_location=dev)
        net.load_state_dict(ck["model"]); net.eval()
        t0 = sync_time()
        with torch.no_grad():
            inp, x0 = make_input(yn[None], ones[None], a[None])
            x = net(inp, a[None], x0)[0]
        secs = sync_time() - t0
        finish(m, knots_to_cube(x), secs, {"checkpoint": "09 runs/mst/ckpt_last.pt", "train_steps": ck["step"],
                                          "validity": "all ones", "input_size": [h, w], "fully_convolutional": True})
    elif m == "inr":
        xy = make_xy(h, w, dev)
        wl = torch.linspace(0.0, 1.0, L, device=dev)
        set_seed(cfg.seed)
        model = build_2d(cfg, sig, dev)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        logf = open(args.out / "inr_train_log.txt", "w")
        fit_frame(model, opt, xy, wl, M601, y, ms, sig, h, w, cfg, cfg.epochs, cfg.learning_rate,
                  cfg.min_learning_rate, lambda s: (logf.write(s + "\n"), logf.flush()))
        secs = sync_time() - t0
        finish(m, render_cube(model, xy, wl), secs, {"config": "08/E1 (FiLM HashGrid, independent field)", "epochs": cfg.epochs,
                                                    "batch": cfg.batch_size, "lr": cfg.learning_rate, "seed": cfg.seed})
    torch.cuda.empty_cache()
print("done")
