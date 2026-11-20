import json, sys
from pathlib import Path
import numpy as np

from insf import paths
G = paths.OUT / "generalization"
HERE = Path(__file__).resolve().parent
CAVE = paths.CAVE
KAIST = paths.KAIST
HARV = paths.HARVARD


def box(a, k):
    h, w, c = a.shape
    return a[:h // k * k, :w // k * k].reshape(h // k, k, w // k, k, c).mean((1, 3))


def scene_list():
    hold = json.loads(paths.HARVARD_HOLDOUT.read_text())["held_out"]
    out = [("harvard", f[:-4]) for f in hold]
    out += [("kaist", f"scene{i:02d}") for i in range(1, 31)]
    out += [("cave", p.name[:-3]) for p in sorted(CAVE.glob("*_ms")) if p.is_dir()]
    return out


def load31(ds, name):
    if ds == "cave":
        from PIL import Image
        d = CAVE / f"{name}_ms" / f"{name}_ms"
        c = np.stack([np.asarray(Image.open(d / f"{name}_ms_{k:02d}.png"), np.float32) for k in range(1, 32)], -1)
        if c.ndim == 4:
            c = c[..., 0, :]
        return c / 65535.0, np.arange(400, 701, 10)
    if ds == "kaist":
        import OpenEXR
        ch = OpenEXR.File(str(KAIST / f"{name}_reflectance.exr")).channels()
        nm = np.arange(420, 721, 10)
        c = np.stack([np.asarray(ch[f"w{n}nm"].pixels, np.float32) for n in nm], -1)
        H, W = c.shape[:2]; r0, c0 = (H - 2560) // 2, (W - 2560) // 2
        return box(c[r0:r0 + 2560, c0:c0 + 2560], 5), nm
    if ds == "harvard":
        from scipy.io import loadmat
        c = loadmat(HARV / f"{name}.mat", variable_names=["ref"])["ref"].astype(np.float32)
        H, W = c.shape[:2]; r0, c0 = (H - 1024) // 2, (W - 1024) // 2
        return box(c[r0:r0 + 1024, c0:c0 + 1024], 2), np.arange(420, 721, 10)
    raise ValueError(ds)


def to601(c31, nm31):
    nm = np.arange(350, 951)
    lo, hi, n = nm31[0], nm31[-1], len(nm31)
    W = np.zeros((n, 601), np.float32)
    for l, x in enumerate(nm):
        xc = min(max(x, lo), hi); f = (xc - lo) / 10.0; k = min(int(f), n - 2); r = f - k
        W[k, l] += 1 - r; W[k + 1, l] += r
    return np.nan_to_num(c31).clip(0) @ W
