#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from insf.config import Config
from insf.data.io import _load_measurement, centered_crop_slices

BANDS_NM = (450, 550, 650, 750, 850)


def load_stack(run: Path, sub: str) -> np.ndarray:
    files = sorted((run / sub).glob("frame_*.npy"))
    return np.stack([np.load(f) for f in files], 0)


def to_display_rgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0, None)
    means = lin.reshape(-1, 3).mean(0)
    lin = lin * (means.mean() / np.maximum(means, 1e-12))
    scale = np.percentile(lin, 99.5)
    return (np.clip(lin / scale, 0, 1) ** (1 / 2.2) * 255).round().astype(np.uint8)


def to_display_gray(stack: np.ndarray, lo_hi=None) -> np.ndarray:
    lo, hi = np.percentile(stack, [0.5, 99.5]) if lo_hi is None else lo_hi
    return (np.clip((stack - lo) / max(hi - lo, 1e-12), 0, 1) * 255).round().astype(np.uint8)


def label(img: np.ndarray, text: str, size: int = 22) -> np.ndarray:
    im = Image.fromarray(img if img.ndim == 3 else np.repeat(img[..., None], 3, -1))
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:
        font = ImageFont.load_default()
    d.rectangle([0, 0, 12 + size * 0.6 * len(text), size + 10], fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 255, 255), font=font)
    return np.asarray(im)


def write_mp4(frames: list[np.ndarray], path: Path, fps: float) -> None:
    import imageio_ffmpeg

    h, w = frames[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(str(path), (w, h), fps=fps, codec="libx264",
                                         pix_fmt_out="yuv420p", quality=8,
                                         macro_block_size=16)
    writer.send(None)
    for f in frames:
        writer.send(np.ascontiguousarray(f))
    writer.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="name=run_dir ...")
    p.add_argument("--frame-dir", type=Path, required=True)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fps", type=float, default=5.7)
    p.add_argument("--show-frames", type=int, nargs="+", default=[0, 24, 48, 72, 97])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = dict(r.split("=", 1) for r in args.runs)

    cfg = Config()
    cfg.crop_size = args.crop_size
    rows, cols, _, _ = centered_crop_slices(cfg)
    paths = sorted(args.frame_dir.glob("*.mat"))
    meas = np.stack([_load_measurement(p, "measurement", rows, cols) for p in paths], 0)
    meas_disp = to_display_gray(meas)
    T = meas.shape[0]

    rgb_disp, band_stacks, metrics = {}, {}, {}
    for name, run in runs.items():
        run = Path(run)
        lin = load_stack(run, "rgb_linear")[:T]
        rgb_disp[name] = to_display_rgb(lin)
        band_stacks[name] = load_stack(run, "bands")[:T]
        out = args.out / name / "rgb_png"
        out.mkdir(parents=True, exist_ok=True)
        for t, img in enumerate(rgb_disp[name]):
            Image.fromarray(img).save(out / f"frame_{t:03d}.png")
        write_mp4([label(f, f"{name}  t={t}") for t, f in enumerate(rgb_disp[name])],
                  args.out / name / f"{name}_rgb.mp4", args.fps)
        with open(run / "metrics_per_frame.csv") as f:
            per = list(csv.DictReader(f))
        summary = json.loads((run / "summary.json").read_text())
        d = rgb_disp[name].astype(np.float32)
        summary["flicker_display_rgb"] = float(np.mean(np.abs(np.diff(d, axis=0))) / d.mean())
        metrics[name] = {"summary": summary, "per_frame": per}

    m = meas_disp.astype(np.float32)
    flicker_meas_disp = float(np.mean(np.abs(np.diff(m, axis=0))) / m.mean())


    names = list(runs)
    comp = []
    for t in range(T):
        panels = [label(meas_disp[t], f"measurement t={t}")] + [label(rgb_disp[n][t], n) for n in names]
        comp.append(np.concatenate(panels, axis=1))
    write_mp4(comp, args.out / "comparison_side_by_side.mp4", args.fps)
    for t in args.show_frames:
        if t < T:
            Image.fromarray(comp[t]).save(args.out / f"comparison_frame_{t:03d}.png")


    for t in args.show_frames:
        if t >= T:
            continue
        grid_rows = []
        for n in names:
            st = band_stacks[n]
            tiles = []
            for b, nm in enumerate(BANDS_NM):
                lo_hi = np.percentile(st[:, ::4, ::4, b], [0.5, 99.5])
                tiles.append(label(to_display_gray(st[t, :, :, b], lo_hi), f"{n} {nm}nm", 18))
            grid_rows.append(np.concatenate(tiles, 1))
        Image.fromarray(np.concatenate(grid_rows, 0)).save(args.out / f"bands_frame_{t:03d}.png")

    combined = {n: metrics[n]["summary"] for n in names}
    combined["_measurement"] = {"flicker_display_gray": flicker_meas_disp}
    (args.out / "metrics_summary.json").write_text(json.dumps(combined, indent=2))
    keys = ["frame", "split", "rmse", "rel_rmse", "rmse_unsaturated", "flicker_cube",
            "flicker_pred_meas", "flicker_meas", "fit_seconds"]
    with open(args.out / "metrics_per_frame_all.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["method"] + keys)
        for n in names:
            for r in metrics[n]["per_frame"]:
                wr.writerow([n] + [r.get(k, "") for k in keys])
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
