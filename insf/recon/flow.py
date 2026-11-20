#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

OFFSETS = (-2, -1, 1, 2)


def to_u8(img, lo, hi):
    return np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def warp(img, flow):
    h, w = img.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img.astype(np.float32), gx + flow[..., 0], gy + flow[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def confidence(flow_st, flow_ts, thr=1.0):
    h, w = flow_st.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    tx, ty = gx + flow_st[..., 0], gy + flow_st[..., 1]
    inside = (tx >= 0) & (tx <= w - 1) & (ty >= 0) & (ty <= h - 1)
    back = np.stack([cv2.remap(flow_ts[..., c], tx, ty, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE) for c in range(2)], -1)
    err = np.linalg.norm(flow_st + back, axis=-1)
    return inside & (err <= thr)


def compute_flows(imgs_u8, preset):
    dis = cv2.DISOpticalFlow_create(preset)
    T = len(imgs_u8)
    h, w = imgs_u8[0].shape
    flows = {d: np.zeros((T, h, w, 2), np.float32) for d in OFFSETS}
    valid = {d: np.zeros(T, bool) for d in OFFSETS}
    for t in range(T):
        for d in OFFSETS:
            s = t + d
            if 0 <= s < T:
                flows[d][t] = dis.calc(imgs_u8[s], imgs_u8[t], None)
                valid[d][t] = True
    return flows, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--preset", default="medium", choices=["ultrafast", "fast", "medium"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    preset = {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
              "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
              "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}[args.preset]

    paths = sorted(args.cube_dir.glob("frame_*.npy"))
    if args.frames:
        paths = paths[: args.frames]
    t0 = time.perf_counter()
    bands = {}
    for nm in (550, 750, 850):
        bands[nm] = np.stack([np.load(p, mmap_mode="r")[:, :, nm - 350].astype(np.float32)
                              for p in paths])
    print(f"loaded {len(paths)} frames in {time.perf_counter()-t0:.1f}s", flush=True)
    b750 = bands[750]
    lo, hi = np.percentile(b750, 0.5), np.percentile(b750, 99.5)
    imgs = [to_u8(b, lo, hi) for b in b750]

    t0 = time.perf_counter()
    flows, valid = compute_flows(imgs, preset)
    flow_s = time.perf_counter() - t0
    print(f"DIS flows in {flow_s:.1f}s", flush=True)

    T = len(paths)
    conf = {d: np.zeros((T,) + imgs[0].shape, bool) for d in OFFSETS}
    for d in OFFSETS:
        for t in range(T):
            if valid[d][t]:
                conf[d][t] = confidence(flows[d][t], flows[-d][t + d])
    for d in OFFSETS:
        np.save(args.out / f"flow_d{d:+d}.npy", flows[d].astype(np.float16))
        np.save(args.out / f"conf_d{d:+d}.npy", conf[d])
        np.save(args.out / f"valid_d{d:+d}.npy", valid[d])


    rows = []
    for d in OFFSETS:
        for t in range(T):
            if not valid[d][t]:
                continue
            s = t + d
            c = conf[d][t]
            mag = np.linalg.norm(flows[d][t], axis=-1)
            row = {"t": t, "d": d, "conf_frac": float(c.mean()),
                   "flow_mag_mean": float(mag.mean()), "flow_mag_p95": float(np.percentile(mag, 95))}
            for nm, b in bands.items():
                pred = warp(b[t], flows[d][t])
                denom = float(np.abs(b[s]).mean())
                row[f"res_unwarped_{nm}"] = float(np.abs(b[s] - b[t]).mean() / denom)
                row[f"res_warped_{nm}"] = float(np.abs(b[s] - pred).mean() / denom)
                row[f"res_warped_conf_{nm}"] = float(np.abs(b[s] - pred)[c].mean() / denom)
                row[f"res_unwarped_conf_{nm}"] = float(np.abs(b[s] - b[t])[c].mean() / denom)
            rows.append(row)
    import csv
    with open(args.out / "flow_metrics.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    summary = {"frames": T, "dis_preset": args.preset, "flow_seconds": flow_s,
               "u8_range_750": [float(lo), float(hi)]}
    for d in OFFSETS:
        for sel_name, sel in (("all", lambda r: True), ("fast40_85", lambda r: 40 <= r["t"] <= 85)):
            rs = [r for r in rows if r["d"] == d and sel(r)]
            key = f"d{d:+d}_{sel_name}"
            summary[key] = {k: float(np.mean([r[k] for r in rs])) for k in rows[0] if k not in ("t", "d")}
    (args.out / "flow_summary.json").write_text(json.dumps(summary, indent=2))
    for k, v in summary.items():
        if isinstance(v, dict):
            print(k, {kk: round(vv, 4) for kk, vv in v.items()
                      if kk.startswith(("res_", "conf", "flow_mag_mean"))})


if __name__ == "__main__":
    main()
