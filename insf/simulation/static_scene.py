#!/usr/bin/env python3
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from insf import paths
from insf.recon.single_frame import Config
from insf.data.io import _load_mask, centered_crop_slices

K, SIG_R, SEED = 4.0, 2.0, 20261020
OUT = paths.OUT / "static_single_frame" / "sim_flowers"; OUT.mkdir(parents=True, exist_ok=True)
C = str(paths.CAVE / "flowers_ms" / "flowers_ms")
cube31 = np.stack([np.asarray(Image.open(f"{C}/flowers_ms_{k:02d}.png"), np.float32) for k in range(1, 32)], -1) / 65535.0
nm31 = np.arange(400, 701, 10); nm = np.arange(350, 951)
W = np.zeros((31, 601), np.float32)
for l, x in enumerate(nm):
    xc = min(max(x, 400), 700); f = (xc - 400) / 10.0; k = min(int(f), 29); r = f - k
    W[k, l] += 1 - r; W[k + 1, l] += r
gt = cube31 @ W
cfg = Config(); cfg.crop_size = 512
rows, cols, r0, c0 = centered_crop_slices(cfg)
mask = _load_mask(paths.MASK, "dataCube", rows, cols)
y0 = (gt * mask).sum(-1)
real = np.load(OUT.parent / "c512" / "measurement.npy")
s = float(np.percentile(real, 99) / np.percentile(y0, 99))
gt *= s; y0 *= s
rng = np.random.default_rng(SEED)
raw = y0 / cfg.measurement_gain
raw = rng.poisson(np.clip(raw, 0, None) * K) / K + rng.normal(0, SIG_R, raw.shape)
raw = np.clip(np.round(raw), 0, 1023).astype(np.float32)
y = raw * np.float32(cfg.measurement_gain)
np.save(OUT / "gt_cube.npy", gt.astype(np.float32))
np.save(OUT / "measurement_sim.npy", y)
meta = {"scene": "CAVE flowers (flowers_ms), 512x512, 31 bands 400:10:700 nm",
        "spectral_extension": "linear interpolation to 1 nm inside 400-700 nm; edge values held for 350-399 and 701-950 nm",
        "scale": s, "target_p99": float(np.percentile(real, 99)), "real_mean": float(real.mean()), "sim_mean": float(y.mean()),
        "noise": {"shot_e_per_DN": K, "read_sigma_DN": SIG_R, "quantisation": "10-bit, clip [0,1023]", "seed": SEED},
        "noiseless_vs_noisy_rmse": float(np.sqrt(((y - y0) ** 2).mean())), "saturated_fraction": float((raw >= 1023).mean()),
        "mask_crop_rows_1based": [r0 + 1, r0 + 512], "mask_crop_cols_1based": [c0 + 1, c0 + 512]}
(OUT / "sim_meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
