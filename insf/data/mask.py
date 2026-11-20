#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
import h5py, numpy as np
from insf import paths
from insf.data.qe import read_qe_xlsx, channel_map

ap = argparse.ArgumentParser()
ap.add_argument("--qe", type=Path, default=paths.QE_XLSX)
ap.add_argument("--out", type=Path, default=paths.MASK)
a = ap.parse_args()
wl, Q = read_qe_xlsx(a.qe)
np.save(a.out.parent / "qe.npy", np.column_stack([wl, Q]))
chan = channel_map().T
a.out.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(a.out, "w") as f:
    d = f.create_dataset("dataCube", shape=(601, 1600, 1200), dtype="f8", chunks=(1, 1600, 1200), compression="gzip", compression_opts=4)
    for k in range(601):
        d[k] = Q[k][chan]
print("wrote", a.out)
