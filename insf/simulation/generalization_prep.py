import json, sys
from pathlib import Path
import numpy as np
from insf.simulation.datasets import G, scene_list, load31, to601
from insf import paths
from insf.recon.single_frame import cie_matrix

cie = cie_matrix(paths.XYZBAR)
(G / "out/gt").mkdir(parents=True, exist_ok=True)
info = []
for ds, name in scene_list():
    p = G / "out/gt" / f"{ds}_{name}.npz"
    c31, nm = load31(ds, name)
    np.savez(p, gt31=c31.astype(np.float16), nm=nm)
    lin = np.clip(to601(c31[::4, ::4], nm) @ cie, 0, None)
    m = lin.reshape(-1, 3).mean(0); lin = lin * (m.mean() / m); lin = np.clip(lin / np.percentile(lin, 99.5), 0, 1) ** (1 / 2.2)
    lum = lin.mean(-1); sel = lum > np.percentile(lum, 10)
    chroma = float((lin.max(-1) - lin.min(-1))[sel].mean())
    info.append({"ds": ds, "name": name, "chroma": chroma, "nm": [int(nm[0]), int(nm[-1])]})
    print(ds, name, round(chroma, 3), c31.shape, flush=True)
rank = {}
for ds in ("kaist", "cave"):
    r = sorted([i for i in info if i["ds"] == ds], key=lambda i: -i["chroma"])
    rank[ds] = [i["name"] for i in r]
A = [("harvard", i["name"], "base") for i in info if i["ds"] == "harvard"]
A += [(ds, n, "base") for ds in ("kaist", "cave") for n in rank[ds][:12]]
S = [(ds, n, c) for ds in ("kaist", "cave") for n in rank[ds][:4] for c in ("noise2x", "noise0.5x", "maskshift")]
B = [(ds, n, "base") for ds in ("kaist", "cave") for n in rank[ds][12:]]
json.dump({"scenes": info, "rank": rank, "queue": [list(q) for q in A + S + B], "phaseA": len(A), "shift": len(S),
           "rule": __doc__}, open(G / "queue.json", "w"), indent=2)
print(len(A), len(S), len(B))
