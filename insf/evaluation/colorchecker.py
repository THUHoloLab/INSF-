#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import colour

ap = argparse.ArgumentParser(); ap.add_argument("runs", type=Path)
ap.add_argument("--lo", type=float, default=420); ap.add_argument("--hi", type=float, default=730)
a = ap.parse_args()
cc = colour.SDS_COLOURCHECKERS["ColorChecker N Ohta"]; names = list(cc.keys())
WL = np.arange(350, 951); sel = (WL >= a.lo) & (WL <= a.hi); w = WL[sel]
ref = np.stack([np.interp(w, cc[n].wavelengths, cc[n].values) for n in names])
ORDER = [(3 - j) * 6 + i for i in range(6) for j in range(4)]
for run in sorted(p for p in a.runs.iterdir() if (p / "spectra.npz").exists()):
    s = np.load(run / "spectra.npz"); Rg = ref[ORDER]
    Rh = s["patch"][:, sel] / np.maximum(s["tile"][sel], 1e-6); Rh *= (Rh * Rg).sum() / (Rh * Rh).sum()
    rmse = np.sqrt(((Rh - Rg) ** 2).mean(1))
    sam = np.degrees(np.arccos(np.clip((Rh * Rg).sum(1) / np.linalg.norm(Rh, axis=1) / np.linalg.norm(Rg, axis=1), -1, 1)))
    print(f"{run.name:28s} reflectance RMSE {rmse.mean():.3f}   spectral angle {sam.mean():5.1f} deg")
