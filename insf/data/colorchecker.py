#!/usr/bin/env python3
import json, re, shutil, sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from insf import paths
from insf.data.qe import load_pattern
PATTERN = load_pattern()

HERE = Path(__file__).resolve().parent; OUT = paths.OUT / "colorchecker"
SCENE = {"k2800": "scene2800k10", "k6500": "scene6500k12"}


def stamp(f):
    h, m, s, ms = [int(x) for x in re.search(r"(\d+)-\s*(\d+)-\s*(\d+)-\s*(\d+)\.raw$", f.name).groups()]
    return h * 3600 + m * 60 + s + ms / 1000


def chan9(a):
    a = a[:1200, :1599]; o = np.zeros((400, 533, 9), np.float32)
    for r in range(3):
        for c in range(3):
            o[:, :, PATTERN[r, c] - 1] = a[r::3, c::3]
    return o


def white_tile(c9):
    b = c9.mean(2); lab, n = ndi.label(b > 0.7 * np.percentile(b, 99.5))
    k = ndi.sum(np.ones_like(b), lab, range(1, n + 1)).argmax() + 1
    return ndi.binary_erosion(lab == k, iterations=6)


F = json.load(open(paths.COLORCHECKER_FRAMES))
for d in ("raw", "tile"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
for key, f in F.items():
    files = sorted((paths.COLORCHECKER_RAW / f["scene"]).glob("*.raw"), key=stamp); i = f["idx"]
    for d in range(-2, 3):
        shutil.copy(files[i + d], OUT / "raw" / f"{key}_d{d:+d}.raw")
    raw = np.fromfile(files[i], dtype="<u2").reshape(1200, 1600).astype(np.float32)
    full = np.zeros((1200, 1600), bool); full[:1200, :1599] = np.kron(white_tile(chan9(raw)), np.ones((3, 3), bool))
    r0, c0 = f["roi"]; np.save(OUT / "tile" / f"{key}.npy", full[r0:r0 + 640, c0:c0 + 640])
    print(key, f["scene"], "frame", i, "tile pixels", int(full[r0:r0 + 640, c0:c0 + 640].sum()))
