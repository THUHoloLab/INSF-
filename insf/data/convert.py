#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
from scipy.io import savemat

src, dst = Path(sys.argv[1]), Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
files = sorted(src.glob("*.qs")) + sorted(src.glob("*.raw"))
for f in files:
    header = 4 if f.suffix == ".qs" else 0
    data = np.fromfile(f, dtype="<u2", count=1200 * 1600, offset=2 * header)
    if data.size != 1200 * 1600:
        print("size mismatch, skipped:", f); continue
    savemat(dst / (f.stem.replace(" ", "0") + ".mat"), {"measurement": data.reshape(1200, 1600).astype(np.float64)})
print(f"converted {len(files)} frames -> {dst}")
