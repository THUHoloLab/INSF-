#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from insf.recon.single_frame import Config, cie_matrix, load_sequence
from insf.evaluation.videos import label, to_display_gray, write_mp4
from PIL import Image

BANDS = (450, 550, 650, 750, 850)
ap = argparse.ArgumentParser()
ap.add_argument("--frame-dir", type=Path, required=True)
ap.add_argument("--mask", type=Path, required=True)
ap.add_argument("--xyzbar", type=Path, required=True)
ap.add_argument("--holdout", type=Path, required=True)
ap.add_argument("--flow-dir", type=Path, required=True)
ap.add_argument("--mc-cubes", type=Path, required=True)
ap.add_argument("--mc-runs", type=Path, nargs="+", required=True)
ap.add_argument("--e1h-cubes", type=Path, required=True)
ap.add_argument("--e1h-run", type=Path, required=True)
ap.add_argument("--e1-cubes", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda")

cfg = Config()
cfg.crop_size = 512
mask_np, frames_np, sat_np, paths, _ = load_sequence(
    argparse.Namespace(frame_dir=args.frame_dir, mask=args.mask, max_frames=None), cfg)
T, H, W = frames_np.shape
L = mask_np.shape[-1]
mask = torch.from_numpy(mask_np.reshape(-1, L)).to(dev)
Y = torch.from_numpy(frames_np.reshape(T, -1)).to(dev)
SAT = torch.from_numpy(sat_np.reshape(T, -1)).to(dev)
HOLD = torch.from_numpy(np.load(args.holdout).reshape(T, -1)).to(dev)
flow1 = np.load(args.flow_dir / "flow_d+1.npy")
conf1 = np.load(args.flow_dir / "conf_d+1.npy")
rgb_mat = torch.from_numpy(cie_matrix(args.xyzbar)).to(dev)
bidx = [nm - 350 for nm in BANDS]
e1h_frames = sorted(int(p.stem[6:]) for p in args.e1h_cubes.glob("frame_*.npy"))
xy = torch.stack(torch.meshgrid((torch.arange(H, device=dev) + 0.5) / H,
                                (torch.arange(W, device=dev) + 0.5) / W, indexing="ij")[::-1], -1).reshape(-1, 2)


def load(d, t):
    return torch.from_numpy(np.load(d / f"frame_{t:03d}.npy").astype(np.float32)).to(dev).reshape(-1, L)


def warp_encode(cube, t_dst, idx, corr=None, gain=1.0, bias=0.0):
    u = torch.from_numpy(flow1[t_dst - 1].reshape(-1, 2).astype(np.float32)).to(dev)[idx]
    if corr is not None:
        g = xy[idx][None, None] * 2 - 1
        u = u + F.grid_sample(corr, g, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T
    c = xy[idx] + u / torch.tensor([W, H], device=dev)
    img = cube.T.reshape(1, L, H, W)
    s = F.grid_sample(img, (c * 2 - 1)[None, None], mode="bilinear", padding_mode="border",
                      align_corners=False)[0, :, 0].T
    return gain * (s * mask[idx]).sum(-1) + bias


def rmse(a, b):
    return float((a - b).square().mean().sqrt())


rows = []
stacks = {k: {"rgb": [], "bands": []} for k in ("mc", "e1", "e1h")}
prev = {}
for t in range(T):
    cubes = {"mc": load(args.mc_cubes, t), "e1": load(args.e1_cubes, t)}
    if t in e1h_frames:
        cubes["e1h"] = load(args.e1h_cubes, t)
    ho, unsat = HOLD[t], ~SAT[t]
    row = {"frame": t, "fast40_85": int(40 <= t <= 85), "e1h_frame": int(t in e1h_frames)}
    for k, c in cubes.items():
        pred = (c * mask).sum(-1)
        row[f"{k}_rmse_holdout"] = rmse(pred[ho], Y[t, ho])
        row[f"{k}_rmse_holdout_unsat"] = rmse(pred[ho & unsat], Y[t, ho & unsat])
        row[f"{k}_rmse_train"] = rmse(pred[~ho], Y[t, ~ho])
        for nm, b in zip(BANDS, bidx):
            row[f"{k}_band{nm}_mean"] = float(c[:, b].mean())
        if k in prev and prev[k][0] == t - 1:
            row[f"{k}_flicker"] = float((c - prev[k][1]).abs().mean() / c.abs().mean())
        stacks[k]["rgb"].append((c @ rgb_mat).reshape(H, W, 3).cpu().numpy())
        stacks[k]["bands"].append(c[:, bidx].reshape(H, W, -1).cpu().numpy())
    for a, b in (("mc", "e1"), ("e1h", "e1"), ("mc", "e1h")):
        if a in cubes and b in cubes:
            for nm, bi in zip(BANDS, bidx):
                row[f"dis_{a}_vs_{b}_{nm}"] = float((cubes[a][:, bi] - cubes[b][:, bi]).abs().mean()
                                                    / cubes[b][:, bi].abs().mean())

    if t >= 1:
        idx = torch.nonzero(ho & torch.from_numpy(conf1[t - 1].reshape(-1)).to(dev)).squeeze(1)
        row["xf_npix"] = int(idx.numel())
        row["xf_e1"] = rmse(warp_encode(prev["e1"][1], t, idx), Y[t, idx])
        ws = torch.load(next(r for r in args.mc_runs if (r / "window_state" / f"frame_{t-1:03d}.pt").exists())
                        / "window_state" / f"frame_{t-1:03d}.pt")
        if "corr.1" in ws:
            row["xf_mc"] = rmse(warp_encode(prev["mc"][1], t, idx, ws["corr.1"].to(dev),
                                            float(ws["log_gain.1"].exp()), float(ws["bias.1"])), Y[t, idx])
            row["xf_mc_dis_only"] = rmse(warp_encode(prev["mc"][1], t, idx), Y[t, idx])
        if (t - 1) in e1h_frames:
            row["xf_e1h"] = rmse(warp_encode(load(args.e1h_cubes, t - 1), t, idx), Y[t, idx])
        row["xf_copy_meas_prev"] = rmse(Y[t - 1, idx], Y[t, idx])
    for k, c in cubes.items():
        prev[k] = (t, c)
    rows.append(row)
    print(t, {k: round(v, 3) for k, v in row.items() if "rmse_holdout" in k and "unsat" not in k or k.startswith("xf_")}, flush=True)

keys = []
for r in rows:
    keys += [k for k in r if k not in keys]
with open(args.out / "metrics_per_frame_mc_vs_e1.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=keys)
    wr.writeheader()
    wr.writerows(rows)


def mean(key, sel):
    v = [r[key] for r in rows if sel(r) and key in r]
    return (float(np.mean(v)), len(v)) if v else (None, 0)


shared = lambda r: r["e1h_frame"]
fast_shared = lambda r: r["e1h_frame"] and r["fast40_85"]
summ = {}
for name, sel in (("shared33", shared), ("shared_fast40_85", fast_shared)):
    d = {}
    for k in ("mc", "e1h"):
        for m in ("rmse_holdout", "rmse_holdout_unsat", "rmse_train"):
            d[f"{k}_{m}"] = mean(f"{k}_{m}", sel)[0]
    d["n_frames"] = mean("mc_rmse_holdout", sel)[1]
    d["mc_wins"] = int(sum(r["mc_rmse_holdout"] < r["e1h_rmse_holdout"] for r in rows if sel(r)))
    d["holdout_reduction_pct"] = 100 * (1 - d["mc_rmse_holdout"] / d["e1h_rmse_holdout"])
    summ[name] = d
summ["mc_all98_rmse_holdout"] = mean("mc_rmse_holdout", lambda r: True)[0]
summ["mc_fast40_85_all_rmse_holdout"] = mean("mc_rmse_holdout", lambda r: r["fast40_85"])[0]
summ["flicker"] = {k: mean(f"{k}_flicker", lambda r: True)[0] for k in ("mc", "e1")}
summ["cross_frame"] = {k: mean(k, lambda r: True) for k in ("xf_e1", "xf_mc", "xf_mc_dis_only", "xf_copy_meas_prev")}
summ["cross_frame_e1h_frames"] = {k: mean(k, lambda r: "xf_e1h" in r) for k in ("xf_e1h", "xf_e1", "xf_mc")}
summ["visible_band_disagreement_shared33"] = {
    pair: {nm: mean(f"dis_{pair}_{nm}", shared)[0] for nm in BANDS}
    for pair in ("e1h_vs_e1", "mc_vs_e1", "mc_vs_e1h")}
summ["band_means_shared33"] = {k: {nm: mean(f"{k}_band{nm}_mean", shared)[0] for nm in BANDS}
                               for k in ("mc", "e1h", "e1")}
(args.out / "summary_mc_vs_e1.json").write_text(json.dumps(summ, indent=2))
print(json.dumps(summ, indent=2))


def to_rgb_disp(stack):
    lin = np.clip(np.stack(stack), 0, None)
    m = lin.reshape(-1, 3).mean(0)
    lin = lin * (m.mean() / np.maximum(m, 1e-12))
    return (np.clip(lin / np.percentile(lin, 99.5), 0, 1) ** (1 / 2.2) * 255).round().astype(np.uint8)

rgb = {k: to_rgb_disp(v["rgb"]) for k, v in stacks.items()}
bands = {k: np.stack(v["bands"]) for k, v in stacks.items()}
meas_disp = to_display_gray(frames_np)
names = {"e1": "E1-100 (all px)", "e1h": "E1-100 holdout", "mc": "MC-window K=5"}
(args.out / "rgb_png").mkdir(exist_ok=True)
(args.out / "bands_png").mkdir(exist_ok=True)
e1h_pos = {t: i for i, t in enumerate(e1h_frames)}
for t in (0, 24, 48, 63, 72, 96):
    panels = [label(meas_disp[t], f"measurement t={t}")] + \
        [label(rgb[k][t if k != "e1h" else e1h_pos[t]], names[k]) for k in ("e1", "e1h", "mc")]
    Image.fromarray(np.concatenate(panels, 1)).save(args.out / "rgb_png" / f"rgb_compare_{t:03d}.png")
    grid = []
    for k in ("e1", "e1h", "mc"):
        st = bands[k]
        i = t if k != "e1h" else e1h_pos[t]
        tiles = []
        for b, nm in enumerate(BANDS):
            lo_hi = np.percentile(bands["mc"][:, ::4, ::4, b], [0.5, 99.5])
            tiles.append(label(to_display_gray(st[i, :, :, b], lo_hi), f"{names[k]} {nm}nm", 16))
        grid.append(np.concatenate(tiles, 1))
    Image.fromarray(np.concatenate(grid, 0)).save(args.out / "bands_png" / f"bands_{t:03d}.png")


fps = 5.7
write_mp4([np.concatenate([label(meas_disp[t], f"measurement t={t}"), label(rgb["e1"][t], names["e1"]),
                           label(rgb["mc"][t], names["mc"])], 1) for t in range(T)],
          args.out / "video_rgb_meas_e1_mc.mp4", fps)
lo_hi = np.percentile(bands["mc"][:, ::4, ::4, 3], [0.5, 99.5])
write_mp4([np.concatenate([label(to_display_gray(bands[k][t, :, :, 3], lo_hi), f"{names[k]} 750nm t={t}")
                           for k in ("e1", "mc")], 1) for t in range(T)],
          args.out / "video_750nm_e1_mc.mp4", fps)
write_mp4([np.concatenate([label(rgb["e1h"][i], f"{names['e1h']} t={t}"), label(rgb["mc"][t], names["mc"])], 1)
           for i, t in enumerate(e1h_frames)], args.out / "video_rgb_e1holdout_mc_33frames.mp4", fps / 3)
np.save(args.out / "mc_bands_all.npy", bands["mc"].astype(np.float16))
print("done")
