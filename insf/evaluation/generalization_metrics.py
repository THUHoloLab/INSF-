#!/usr/bin/env python3
import csv, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import sys
from insf import paths
G = paths.OUT / "generalization"; dev = torch.device("cuda")
METHODS = ["twist", "admm", "mst09", "mst_ho", "inr"]


def ssim(a, b):
    g = torch.exp(-((torch.arange(11, device=dev) - 5.0) ** 2) / (2 * 1.5 ** 2)); g = g / g.sum()
    w = (g[:, None] * g[None])[None, None].expand(a.shape[0], 1, 11, 11)
    f = lambda x: F.conv2d(x[None], w, groups=a.shape[0])[0]
    c1, c2 = 1e-4, 9e-4; ma, mb = f(a), f(b)
    va, vb, cab = f(a * a) - ma ** 2, f(b * b) - mb ** 2, f(a * b) - ma * mb
    return float((((2 * ma * mb + c1) * (2 * cab + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))).mean())


def psnr(a, b):
    return float(10 * torch.log10(1.0 / (a - b).square().mean()))


def disp(lin, gains, expo):
    lin = (lin.clamp_min(0) * gains / expo).clamp(0, 1)
    return torch.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin.clamp_min(1e-12) ** (1 / 2.4) - 0.055)


L = lambda p: torch.from_numpy(np.load(p).astype(np.float32)).to(dev)
rows = []
for d in sorted((G / "out/runs").iterdir()):
    if not (d / "DONE").exists():
        continue
    meta = json.loads((d / "meta.json").read_text())
    gt = L(d / "gt_bands.npy"); gmax = gt.max(); gn = (gt / gmax).permute(2, 0, 1)
    lg = L(d / "gt_rgb.npy").clamp_min(0); m = lg.reshape(-1, 3).mean(0); gains = m.mean() / m
    expo = float(torch.quantile((lg * gains).reshape(-1)[::3], 0.995)); rg = disp(lg, gains, expo)
    for k in METHODS:
        if not (d / f"{k}_bands.npy").exists():
            continue
        c = L(d / f"{k}_bands.npy"); cn = (c / gmax).permute(2, 0, 1)
        num = (c * gt).sum(-1); den = c.norm(dim=-1) * gt.norm(dim=-1); ok = den > 1e-12
        sam = float(torch.rad2deg(torch.arccos((num[ok] / den[ok]).clamp(-1, 1))).mean())
        r = disp(L(d / f"{k}_rgb.npy"), gains, expo)
        rows.append({"ds": meta["ds"], "scene": meta["name"], "cond": meta["cond"], "method": k,
                     "psnr_rgb": psnr(r, rg), "ssim_rgb": ssim(r.permute(2, 0, 1), rg.permute(2, 0, 1)),
                     "psnr_cube": psnr(cn, gn), "ssim_cube": ssim(cn, gn), "sam_deg": sam,
                     "rmse_measurement": meta["methods"][k]["rmse_measurement"], "noise_rmse": meta["noise_rmse"],
                     "seconds": meta["methods"][k]["seconds"]})
with open(G / "out/metrics_items.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
summ = {}
for key in sorted({(r["ds"], r["cond"]) for r in rows}):
    sub = [r for r in rows if (r["ds"], r["cond"]) == key]
    scenes = sorted({r["scene"] for r in sub})
    common = [s for s in scenes if all(any(r["scene"] == s and r["method"] == k for r in sub) for k in METHODS)]
    ent = {"n_scenes_all_methods": len(common), "scenes": common}
    for k in METHODS:
        v = [r for r in sub if r["method"] == k and r["scene"] in common]
        if v:
            ent[k] = {mtr: [float(np.mean([r[mtr] for r in v])), float(np.std([r[mtr] for r in v]))]
                      for mtr in ("psnr_rgb", "ssim_rgb", "psnr_cube", "ssim_cube", "sam_deg", "rmse_measurement", "seconds")}
            ent[k]["inr_wins_psnr_rgb"] = None
    if "inr" in ent:
        for k in METHODS[:-1]:
            if k in ent:
                pr = {r["scene"]: r["psnr_rgb"] for r in sub if r["method"] == k and r["scene"] in common}
                pi = {r["scene"]: r["psnr_rgb"] for r in sub if r["method"] == "inr" and r["scene"] in common}
                ent[k]["inr_better_psnr_rgb_scenes"] = sum(pi[s] > pr[s] for s in common)
                pr = {r["scene"]: r["psnr_cube"] for r in sub if r["method"] == k and r["scene"] in common}
                pi = {r["scene"]: r["psnr_cube"] for r in sub if r["method"] == "inr" and r["scene"] in common}
                ent[k]["inr_better_psnr_cube_scenes"] = sum(pi[s] > pr[s] for s in common)
    summ[f"{key[0]}|{key[1]}"] = ent
(G / "out/summary.json").write_text(json.dumps(summ, indent=2))
for k, e in summ.items():
    print(k, e["n_scenes_all_methods"], {m: [round(e[m]["psnr_rgb"][0], 2), round(e[m]["psnr_cube"][0], 2), round(e[m]["sam_deg"][0], 1)] for m in METHODS if m in e})
