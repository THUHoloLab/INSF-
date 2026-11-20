#!/usr/bin/env python3

from __future__ import annotations
import sys
from pathlib import Path
from insf import paths

import argparse
import datetime as dt
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import savemat

from insf.config import Config
from insf.data.io import _load_mask, _load_measurement, centered_crop_slices
from insf.models.field import SpatialSpectralField, count_parameters
from insf.recon.single_image import charbonnier_loss, cosine_learning_rate, make_xy_grid, set_seed


TIME_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})--"
    r"(?P<hour>\d{2})-(?P<minute>\d{2})-(?P<second>\d{2})-"
    r"(?P<millis>\d{3})\.mat$"
)


@dataclass
class FrameInfo:
    index: int
    path: Path
    timestamp: str
    timestamp_seconds: float
    dt_seconds: float
    mean: float
    median: float
    p99: float
    minimum: float
    maximum: float
    saturation_fraction: float
    zero_fraction: float
    quality: str


@dataclass
class MultiFrameData:
    mask: np.ndarray
    measurements: np.ndarray
    frames: list[FrameInfo]
    row_start0: int
    col_start0: int
    measurement_scale: float
    signal_scale: float
    train_count: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (*self.mask.shape[:2], self.mask.shape[2])


def parse_timestamp(path: Path) -> dt.datetime:
    match = TIME_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse timestamp from {path.name}")
    values = match.groupdict()
    return dt.datetime.strptime(
        f"{values['date']} {values['hour']}:{values['minute']}:{values['second']}",
        "%Y-%m-%d %H:%M:%S",
    ).replace(microsecond=int(values["millis"]) * 1000)


def list_frame_paths(directory: Path) -> list[tuple[dt.datetime, Path]]:
    paths = [(parse_timestamp(path), path) for path in directory.glob("*.mat")]
    if not paths:
        raise FileNotFoundError(f"No timestamped MAT frames in {directory}")
    return sorted(paths)


def _stats(array: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    return (
        float(np.mean(array)),
        float(np.median(array)),
        float(np.percentile(array, 99)),
        float(np.min(array)),
        float(np.max(array)),
        float(np.mean(array >= 1023.0)),
        float(np.mean(array == 0.0)),
    )


def load_multiframe_data(cfg: Config, args: argparse.Namespace) -> MultiFrameData:
    rows, cols, row_start0, col_start0 = centered_crop_slices(cfg)
    mask = _load_mask(Path(cfg.mask_path), cfg.mask_key, rows, cols)
    all_paths = list_frame_paths(Path(args.frame_dir))
    if args.end_frame is not None:
        all_paths = all_paths[: args.end_frame + 1]
    if args.frame_stride > 1:
        all_paths = all_paths[:: args.frame_stride]
    if args.max_frames is not None:
        all_paths = all_paths[: args.max_frames]
    if len(all_paths) < 2:
        raise ValueError("At least two frames are needed for a multi-frame run.")

    arrays: list[np.ndarray] = []
    frame_infos: list[FrameInfo] = []
    previous_time: dt.datetime | None = None
    for index, (timestamp, path) in enumerate(all_paths):
        array = _load_measurement(path, cfg.measurement_key, rows, cols)
        array = np.asarray(array * np.float32(cfg.measurement_gain), dtype=np.float32)
        if array.shape != mask.shape[:2]:
            raise ValueError(
                f"Spatial mismatch for {path.name}: measurement={array.shape}, "
                f"mask={mask.shape[:2]}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite values in {path}")
        arrays.append(np.ascontiguousarray(array))
        stats = _stats(array)
        frame_infos.append(
            FrameInfo(
                index=index,
                path=path,
                timestamp=timestamp.isoformat(timespec="milliseconds"),
                timestamp_seconds=timestamp.timestamp(),
                dt_seconds=(
                    0.0
                    if previous_time is None
                    else (timestamp - previous_time).total_seconds()
                ),
                mean=stats[0],
                median=stats[1],
                p99=stats[2],
                minimum=stats[3],
                maximum=stats[4],
                saturation_fraction=stats[5],
                zero_fraction=stats[6],
                quality="ok",
            )
        )
        previous_time = timestamp

    measurements = np.stack(arrays, axis=0)
    train_count = args.train_count or len(frame_infos)
    train_count = min(train_count, len(frame_infos))
    if train_count < 1:
        raise ValueError("train_count must be positive.")


    for frame in frame_infos:
        if frame.maximum > args.bad_max or frame.zero_fraction > args.bad_zero_fraction:
            frame.quality = "reject_bad_value"
    rejected = [frame.index for frame in frame_infos if frame.quality != "ok"]
    if rejected:
        raise ValueError(
            "Selected window contains rejected frames "
            f"{rejected}; reduce --end-frame or explicitly clean the data."
        )

    reference = np.median(measurements[:train_count], axis=0)
    measurement_scale = max(float(np.percentile(np.abs(measurements[:train_count]), 99)), 1e-6)
    mask_sum = mask.sum(axis=-1)
    valid = mask_sum > 1e-6
    if not np.any(valid):
        raise ValueError("Mask has no nonzero spectral throughput in the selected ROI.")
    signal_scale = max(float(np.median(np.abs(reference[valid] / mask_sum[valid]))), 1e-6)

    return MultiFrameData(
        mask=mask,
        measurements=measurements,
        frames=frame_infos,
        row_start0=row_start0,
        col_start0=col_start0,
        measurement_scale=measurement_scale,
        signal_scale=signal_scale,
        train_count=train_count,
    )


def fit_affine(reference: np.ndarray, target: np.ndarray) -> tuple[float, float, float, float]:
    ref = reference.reshape(-1).astype(np.float64)
    tar = target.reshape(-1).astype(np.float64)
    keep = np.isfinite(ref) & np.isfinite(tar)
    ref = ref[keep]
    tar = tar[keep]

    low, high = np.percentile(tar, [1, 99])
    keep = (tar >= low) & (tar <= high)
    design = np.column_stack((ref[keep], np.ones(int(keep.sum()))))
    gain, bias = np.linalg.lstsq(design, tar[keep], rcond=None)[0]
    residual = gain * ref + bias - tar
    return float(gain), float(bias), float(np.mean(np.abs(residual))), float(
        np.sqrt(np.mean(residual**2))
    )


def spatial_tv_loss(
    model: SpatialSpectralField,
    full_xy: torch.Tensor,
    wavelengths: torch.Tensor,
    height: int,
    width: int,
    signal_scale: float,
    epsilon: float,
) -> torch.Tensor:
    band = wavelengths[torch.randint(wavelengths.numel(), (1,), device=full_xy.device)]
    image = model(full_xy, band).reshape(height, width)
    horizontal = image[:, 1:] - image[:, :-1]
    vertical = image[1:, :] - image[:-1, :]
    return 0.5 * (
        charbonnier_loss(horizontal / signal_scale, epsilon)
        + charbonnier_loss(vertical / signal_scale, epsilon)
    )


def make_run_dir(root: Path) -> Path:
    path = root / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path.mkdir(parents=True, exist_ok=False)
    return path


def frame_payload(frame: FrameInfo) -> dict:
    return {
        "index": frame.index,
        "path": str(frame.path),
        "timestamp": frame.timestamp,
        "timestamp_seconds": frame.timestamp_seconds,
        "dt_seconds": frame.dt_seconds,
        "mean": frame.mean,
        "median": frame.median,
        "p99": frame.p99,
        "minimum": frame.minimum,
        "maximum": frame.maximum,
        "saturation_fraction": frame.saturation_fraction,
        "zero_fraction": frame.zero_fraction,
        "quality": frame.quality,
    }


def evaluate_affine(
    base_prediction: np.ndarray,
    measurements: np.ndarray,
    indices: np.ndarray,
) -> list[dict]:
    reports = []
    for index in indices.tolist():
        gain, bias, mae, rmse = fit_affine(base_prediction, measurements[index])
        calibrated = gain * base_prediction + bias
        target = measurements[index]
        reports.append(
            {
                "frame_index": int(index),
                "fit_gain": gain,
                "fit_bias": bias,
                "mae": mae,
                "rmse": rmse,
                "relative_rmse": float(rmse / max(float(np.std(target)), 1e-6)),
                "calibrated_mean": float(np.mean(calibrated)),
                "target_mean": float(np.mean(target)),
            }
        )
    return reports


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
        output[start:stop] = (spectrum * mask[start:stop]).sum(dim=-1).cpu().numpy()
    return output


def train_model(
    model: SpatialSpectralField,
    optimizer: torch.optim.Optimizer,
    frame_log_gain: torch.Tensor,
    frame_bias: torch.Tensor,
    xy: torch.Tensor,
    wavelengths: torch.Tensor,
    mask: torch.Tensor,
    measurements: torch.Tensor,
    data: MultiFrameData,
    cfg: Config,
    args: argparse.Namespace,
    start_epoch: int = 0,
) -> list[dict]:
    num_pixels = xy.shape[0]
    max_steps = num_pixels // cfg.batch_size
    steps = min(max_steps, args.steps_per_epoch or max_steps)
    if steps < 1:
        raise ValueError("batch_size is larger than the number of pixels.")
    full_xy = xy
    history: list[dict] = []
    train_frames = torch.arange(data.train_count, device=xy.device)
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        learning_rate = cosine_learning_rate(cfg, epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        permutation = torch.randperm(num_pixels, device=xy.device)
        sums = {"total": 0.0, "data": 0.0, "spectral_tv": 0.0, "spatial_tv": 0.0}
        for step in range(steps):
            indices = permutation[step * cfg.batch_size : (step + 1) * cfg.batch_size]
            optimizer.zero_grad(set_to_none=True)
            spectrum = model(xy[indices], wavelengths)
            base = (spectrum * mask[indices]).sum(dim=-1)
            gains = frame_log_gain[train_frames].exp()[:, None]
            biases = frame_bias[train_frames, None]
            prediction = gains * base[None, :] + biases
            target = measurements[train_frames][:, indices]
            data_loss = F.smooth_l1_loss(
                prediction / data.measurement_scale,
                target / data.measurement_scale,
                beta=cfg.huber_beta,
            )
            spectral_tv = charbonnier_loss(
                torch.diff(spectrum, dim=-1) / data.signal_scale,
                cfg.tv_epsilon,
            )
            if args.spatial_tv_every and step % args.spatial_tv_every == 0:
                spatial_tv = spatial_tv_loss(
                    model,
                    full_xy,
                    wavelengths,
                    data.mask.shape[0],
                    data.mask.shape[1],
                    data.signal_scale,
                    cfg.tv_epsilon,
                )
            else:
                spatial_tv = torch.zeros((), device=xy.device)
            gain_prior = frame_log_gain[: data.train_count].square().mean()
            bias_prior = (frame_bias[: data.train_count] / data.measurement_scale).square().mean()
            if data.train_count > 1:
                gain_smooth = torch.diff(frame_log_gain[: data.train_count]).square().mean()
            else:
                gain_smooth = torch.zeros((), device=xy.device)
            total = (
                data_loss
                + cfg.spectral_tv_weight * spectral_tv
                + cfg.spatial_tv_weight * spatial_tv
                + args.gain_prior * gain_prior
                + args.bias_prior * bias_prior
                + args.gain_smooth * gain_smooth
            )
            total.backward()
            optimizer.step()
            sums["total"] += float(total.detach())
            sums["data"] += float(data_loss.detach())
            sums["spectral_tv"] += float(spectral_tv.detach())
            sums["spatial_tv"] += float(spatial_tv.detach())
        means = {name: value / steps for name, value in sums.items()}
        means.update({"epoch": epoch + 1, "lr": learning_rate})
        history.append(means)
        print(
            f"Epoch {epoch + 1:03d}/{cfg.epochs}: lr={learning_rate:.3g}, "
            f"total={means['total']:.6g}, data={means['data']:.6g}, "
            f"spectral_tv={means['spectral_tv']:.6g}, "
            f"spatial_tv={means['spatial_tv']:.6g}",
            flush=True,
        )
    return history


def save_preview(cube: np.ndarray, path: Path, band: int) -> None:
    image = cube[:, :, min(max(band - 1, 0), cube.shape[2] - 1)]
    low, high = np.percentile(image, [1, 99])
    normalized = np.clip((image - low) / max(high - low, 1e-8), 0.0, 1.0)
    plt.imsave(path, normalized, cmap="gray", vmin=0.0, vmax=1.0)


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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--spatial-tv-every", type=int, default=8)
    parser.add_argument("--gain-prior", type=float, default=1e-3)
    parser.add_argument("--bias-prior", type=float, default=1e-4)
    parser.add_argument("--gain-smooth", type=float, default=1e-3)
    parser.add_argument("--bad-max", type=float, default=4096.0)
    parser.add_argument("--bad-zero-fraction", type=float, default=0.01)
    parser.add_argument("--measurement-gain", type=float, default=0.5)
    parser.add_argument("--save-cube", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.crop_size = min(args.crop_size, 32)
        args.end_frame = min(args.end_frame, 7)
        args.max_frames = min(args.max_frames or 8, 8)
        args.train_count = min(args.train_count, 6)
        args.epochs = 1
        args.steps_per_epoch = 2
        args.batch_size = 128
        args.spatial_tv_every = 1

    cfg = Config()
    cfg.mask_path = Path(args.mask).resolve()
    cfg.crop_size = args.crop_size
    cfg.original_roi_size = args.original_roi_size
    cfg.original_roi_start_matlab = tuple(args.roi_start)
    cfg.measurement_gain = args.measurement_gain
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.output_root = Path(args.output_root).resolve()
    cfg.save_every = max(args.epochs, 1)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for tiny-cuda-nn multi-frame training.")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    set_seed(cfg.seed)
    device = torch.device("cuda")

    data = load_multiframe_data(cfg, args)
    print(
        f"Frames: {len(data.frames)} (train={data.train_count}, "
        f"validation={len(data.frames) - data.train_count})",
        flush=True,
    )
    print(
        f"ROI: {data.mask.shape[0]}x{data.mask.shape[1]}, bands={data.mask.shape[2]}, "
        f"measurement_scale={data.measurement_scale:.6g}, "
        f"signal_scale={data.signal_scale:.6g}",
        flush=True,
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    height, width, bands = data.mask.shape
    xy = make_xy_grid(height, width, device)
    wavelengths = torch.linspace(0.0, 1.0, bands, device=device)
    mask = torch.from_numpy(data.mask.reshape(-1, bands)).to(device)
    measurements = torch.from_numpy(data.measurements).reshape(len(data.frames), -1).to(device)

    model = SpatialSpectralField(
        num_spectral_knots=cfg.num_spectral_knots,
        output_scale=data.signal_scale,
        n_levels=cfg.n_levels,
        n_features_per_level=cfg.n_features_per_level,
        log2_hashmap_size=cfg.log2_hashmap_size,
        base_resolution=cfg.base_resolution,
        per_level_scale=cfg.per_level_scale,
        hidden_dim=cfg.hidden_dim,
        hidden_layers=cfg.hidden_layers,
    ).to(device)
    frame_log_gain = torch.nn.Parameter(torch.zeros(len(data.frames), device=device))
    frame_bias = torch.nn.Parameter(torch.zeros(len(data.frames), device=device))
    start_epoch = int(args.start_epoch)
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        frame_log_gain.data.copy_(checkpoint["frame_log_gain"].to(device))
        frame_bias.data.copy_(checkpoint["frame_bias"].to(device))
        print(f"Resumed model parameters from {args.resume}", flush=True)
    if start_epoch < 0 or start_epoch >= cfg.epochs:
        raise ValueError(
            f"start_epoch={start_epoch} must be in [0, {cfg.epochs - 1}]"
        )
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [frame_log_gain, frame_bias],
        lr=cfg.learning_rate,
    )
    print(f"Trainable parameters: {count_parameters(model):,} + {2 * len(data.frames)} frame parameters", flush=True)

    run_dir = None if args.smoke_test else make_run_dir(cfg.output_root)
    if run_dir is not None:
        metadata = {
            "mode": "shared_hsi_multiframe_move1",
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "config": cfg.to_dict(),
            "args": {key: str(value) for key, value in vars(args).items()},
            "shape": list(data.mask.shape),
            "train_count": data.train_count,
            "mask_path": str(cfg.mask_path),
            "frames": [frame_payload(frame) for frame in data.frames],
        }
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Run directory: {run_dir}", flush=True)

    history = train_model(
        model,
        optimizer,
        frame_log_gain,
        frame_bias,
        xy,
        wavelengths,
        mask,
        measurements,
        data,
        cfg,
        args,
        start_epoch=start_epoch,
    )
    if args.smoke_test:
        print("Multi-frame smoke test passed.", flush=True)
        return

    print("Rendering shared HSI and measurement prediction...", flush=True)
    cube = None
    if args.save_cube:
        from insf.recon.single_image import reconstruct_cube

        cube = reconstruct_cube(
            model,
            xy,
            wavelengths,
            height,
            width,
            cfg.reconstruction_pixel_batch,
            cfg.reconstruction_spectral_batch,
        )
        savemat(
            run_dir / "reconstruction_result.mat",
            {
                "hsi_pred": cube,
                "crop_size": np.int32(cfg.crop_size),
                "start_x": np.int32(data.row_start0 + 1),
                "start_y": np.int32(data.col_start0 + 1),
                "end_x": np.int32(data.row_start0 + height),
                "end_y": np.int32(data.col_start0 + width),
            },
            do_compression=cfg.compress_mat,
        )
        save_preview(cube, run_dir / "preview_band_150.png", cfg.preview_band)

    base_prediction = render_measurement(model, xy, wavelengths, mask, cfg.reconstruction_pixel_batch)
    base_prediction = base_prediction.reshape(height, width)
    train_indices = np.arange(data.train_count)
    val_indices = np.arange(data.train_count, len(data.frames))
    train_reports = evaluate_affine(base_prediction, data.measurements, train_indices)
    val_reports = evaluate_affine(base_prediction, data.measurements, val_indices)
    temporal_median = np.median(data.measurements[: data.train_count], axis=0)
    median_mae = float(np.mean(np.abs(base_prediction - temporal_median)))
    raw_temporal_std = float(np.median(np.std(data.measurements[: data.train_count], axis=0)))
    frame_gains = frame_log_gain.detach().cpu().numpy().tolist()
    frame_biases = frame_bias.detach().cpu().numpy().tolist()
    result = {
        "history": history,
        "learned_log_gain": frame_gains,
        "learned_bias": frame_biases,
        "train_affine_reports": train_reports,
        "validation_affine_reports": val_reports,
        "validation_mean_rmse": float(np.mean([item["rmse"] for item in val_reports])) if val_reports else None,
        "validation_mean_relative_rmse": float(np.mean([item["relative_rmse"] for item in val_reports])) if val_reports else None,
        "raw_train_temporal_median_std": raw_temporal_std,
        "base_prediction_vs_train_temporal_median_mae": median_mae,
        "base_prediction_range": [float(base_prediction.min()), float(base_prediction.max())],
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "model": model.state_dict(),
            "frame_log_gain": frame_log_gain.detach().cpu(),
            "frame_bias": frame_bias.detach().cpu(),
            "config": cfg.to_dict(),
            "args": {key: str(value) for key, value in vars(args).items()},
        },
        run_dir / "checkpoint.pt",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved run: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
