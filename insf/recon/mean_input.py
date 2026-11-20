#!/usr/bin/env python3

from __future__ import annotations
import sys
from pathlib import Path
from insf import paths

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.io import savemat

from insf.config import Config
from insf.data.io import ReconstructionData
from insf.models.field import SpatialSpectralField, count_parameters
from insf.recon.single_image import (
    create_run_directory,
    make_xy_grid,
    reconstruct_cube,
    save_result,
    set_seed,
    train,
)
from insf.recon.shared_field import evaluate_affine, load_multiframe_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=800)
    parser.add_argument("--original-roi-size", type=int, default=1000)
    parser.add_argument("--roi-start", type=int, nargs=2, default=(200, 200))
    parser.add_argument("--end-frame", type=int, default=103)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--train-count", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--measurement-gain", type=float, default=0.5)
    parser.add_argument("--bad-max", type=float, default=4096.0)
    parser.add_argument("--bad-zero-fraction", type=float, default=0.01)
    parser.add_argument("--save-cube", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def render_measurement(
    model: SpatialSpectralField,
    xy: torch.Tensor,
    wavelengths: torch.Tensor,
    mask: torch.Tensor,
    pixel_batch: int,
) -> np.ndarray:
    model.eval()
    output = np.empty(xy.shape[0], dtype=np.float32)
    for start in range(0, xy.shape[0], pixel_batch):
        stop = min(start + pixel_batch, xy.shape[0])
        spectrum = model(xy[start:stop], wavelengths)
        output[start:stop] = (
            spectrum * mask[start:stop]
        ).sum(dim=-1).cpu().numpy()
    return output


def metric_summary(
    name: str,
    prediction: np.ndarray,
    measurements: np.ndarray,
    indices: np.ndarray,
) -> dict:
    reports = evaluate_affine(prediction, measurements, indices)
    return {
        "name": name,
        "mean_rmse": float(np.mean([item["rmse"] for item in reports])),
        "mean_relative_rmse": float(
            np.mean([item["relative_rmse"] for item in reports])
        ),
        "per_frame": reports,
    }


def main() -> None:
    args = parse_args()
    cfg = Config()
    cfg.mask_path = args.mask.resolve()
    cfg.crop_size = args.crop_size
    cfg.original_roi_size = args.original_roi_size
    cfg.original_roi_start_matlab = tuple(args.roi_start)
    cfg.measurement_gain = args.measurement_gain
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.output_root = args.output_root.resolve()
    cfg.save_every = max(1, min(10, args.epochs))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for tiny-cuda-nn training.")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    set_seed(cfg.seed)
    device = torch.device("cuda")

    loader_args = SimpleNamespace(
        frame_dir=args.frame_dir.resolve(),
        end_frame=args.end_frame,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        train_count=args.train_count,
        bad_max=args.bad_max,
        bad_zero_fraction=args.bad_zero_fraction,
    )
    sequence = load_multiframe_data(cfg, loader_args)
    train_mean = np.mean(sequence.measurements[: sequence.train_count], axis=0).astype(
        np.float32
    )
    mask_sum = sequence.mask.sum(axis=-1)
    valid = mask_sum > 1e-6
    measurement_scale = max(float(np.percentile(np.abs(train_mean), 99)), 1e-6)
    signal_scale = max(
        float(np.median(np.abs(train_mean[valid] / mask_sum[valid]))), 1e-6
    )
    mean_data = ReconstructionData(
        mask=sequence.mask,
        measurement=train_mean,
        row_start0=sequence.row_start0,
        col_start0=sequence.col_start0,
        measurement_scale=measurement_scale,
        signal_scale=signal_scale,
    )

    height, width, bands = mean_data.shape
    xy = make_xy_grid(height, width, device)
    wavelengths = torch.linspace(0.0, 1.0, bands, device=device)
    mask = torch.from_numpy(mean_data.mask.reshape(-1, bands)).to(device)
    measurement = torch.from_numpy(train_mean.reshape(-1)).to(device)
    model = SpatialSpectralField(
        num_spectral_knots=cfg.num_spectral_knots,
        output_scale=mean_data.signal_scale,
        n_levels=cfg.n_levels,
        n_features_per_level=cfg.n_features_per_level,
        log2_hashmap_size=cfg.log2_hashmap_size,
        base_resolution=cfg.base_resolution,
        per_level_scale=cfg.per_level_scale,
        hidden_dim=cfg.hidden_dim,
        hidden_layers=cfg.hidden_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    run_dir = create_run_directory(cfg)

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"Frames: {len(sequence.frames)} (train={sequence.train_count}, "
        f"held-out={len(sequence.frames) - sequence.train_count})",
        flush=True,
    )
    print(
        f"Mean-input ROI: {height}x{width}x{bands}, "
        f"measurement_scale={measurement_scale:.6g}, "
        f"signal_scale={signal_scale:.6g}",
        flush=True,
    )
    print(f"Trainable parameters: {count_parameters(model):,}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)

    metadata = {
        "mode": "mean_then_single_inr_move1",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "config": cfg.to_dict(),
        "args": {key: str(value) for key, value in vars(args).items()},
        "shape": list(mean_data.shape),
        "train_count": sequence.train_count,
        "frames": [
            {
                "index": frame.index,
                "path": str(frame.path),
                "timestamp": frame.timestamp,
                "quality": frame.quality,
            }
            for frame in sequence.frames
        ],
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    savemat(
        run_dir / "training_mean_measurement.mat",
        {"measurement_mean": train_mean},
        do_compression=True,
    )

    train(
        model,
        optimizer,
        xy,
        wavelengths,
        mask,
        measurement,
        mean_data,
        cfg,
        run_dir,
        first_epoch=0,
    )

    print("Rendering HSI and held-out measurement prediction...", flush=True)
    cube = reconstruct_cube(
        model,
        xy,
        wavelengths,
        height,
        width,
        cfg.reconstruction_pixel_batch,
        cfg.reconstruction_spectral_batch,
    )
    save_result(run_dir, cube, mean_data, cfg)
    base_prediction = render_measurement(
        model, xy, wavelengths, mask, cfg.reconstruction_pixel_batch
    ).reshape(height, width)
    train_indices = np.arange(sequence.train_count)
    val_indices = np.arange(sequence.train_count, len(sequence.frames))

    candidates = [
        metric_summary("first_frame", sequence.measurements[0], sequence.measurements, val_indices),
        metric_summary("temporal_mean", train_mean, sequence.measurements, val_indices),
        metric_summary(
            "temporal_median",
            np.median(sequence.measurements[: sequence.train_count], axis=0),
            sequence.measurements,
            val_indices,
        ),
        metric_summary("mean_then_single_inr", base_prediction, sequence.measurements, val_indices),
    ]
    result = {
        "mode": "mean_then_single_inr_move1",
        "training_metric_on_mean_input": metric_summary(
            "mean_then_single_inr_train", base_prediction, sequence.measurements, train_indices
        ),
        "validation_candidates": candidates,
        "prediction_range": [float(base_prediction.min()), float(base_prediction.max())],
        "mean_input_range": [float(train_mean.min()), float(train_mean.max())],
        "cube_shape": list(cube.shape),
        "run_dir": str(run_dir),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved run: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
