import sys
from pathlib import Path
import numpy as np
from PIL import Image
import sys
from insf import paths
G = paths.OUT / "generalization"


def disp(lin, gains, expo):
    lin = np.clip(np.clip(lin, 0, None) * gains / expo, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.maximum(lin, 1e-12) ** (1 / 2.4) - 0.055)


for item in sys.argv[1:]:
    d = G / "out/runs" / item; o = G / "out/png" / item; o.mkdir(parents=True, exist_ok=True)
    lg = np.clip(np.load(d / "gt_rgb.npy"), 0, None); m = lg.reshape(-1, 3).mean(0); gains = m.mean() / m
    expo = float(np.quantile((lg * gains).reshape(-1)[::3], 0.995))
    for k in ("gt", "twist", "admm", "mst09", "mst_ho", "inr"):
        Image.fromarray((disp(np.load(d / f"{k}_rgb.npy"), gains, expo) * 255).round().astype(np.uint8)).save(o / f"{k}_full.png")
    y = np.load(d / "y.npy")
    Image.fromarray((np.clip(y / np.percentile(y, 99.5), 0, 1) * 255).round().astype(np.uint8)).convert("RGB").save(o / "measurement_full.png")
    print(item)
