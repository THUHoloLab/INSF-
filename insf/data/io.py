from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

import time

from insf.config import Config
from insf.forward import SATURATION


EPS = 1e-6


@dataclass
class ReconstructionData:

    mask: np.ndarray
    measurement: np.ndarray
    row_start0: int
    col_start0: int
    measurement_scale: float
    signal_scale: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.mask.shape


def centered_crop_slices(
    cfg: Config,
) -> tuple[slice, slice, int, int]:
    if cfg.crop_size < 1:
        raise ValueError(f"crop_size must be positive, got {cfg.crop_size}.")
    if cfg.crop_size > cfg.original_roi_size:
        raise ValueError(
            f"crop_size={cfg.crop_size} exceeds the original "
            f"{cfg.original_roi_size}x{cfg.original_roi_size} ROI. This program "
            "takes a centered sub-crop from that ROI, so changing crop_size to "
            "1024 alone is not valid; redefine the ROI and its start position first."
        )

    offset = (cfg.original_roi_size - cfg.crop_size) // 2
    row_start0 = cfg.original_roi_start_matlab[0] - 1 + offset
    col_start0 = cfg.original_roi_start_matlab[1] - 1 + offset
    rows = slice(row_start0, row_start0 + cfg.crop_size)
    cols = slice(col_start0, col_start0 + cfg.crop_size)
    return rows, cols, row_start0, col_start0


def _load_mask(path: Path, key: str, rows: slice, cols: slice) -> np.ndarray:
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as file:
            dataset = file[key]
            if dataset.ndim != 3:
                raise ValueError(f"{key!r} must be 3D, got {dataset.shape}.")


            mask = np.asarray(dataset[:, cols, rows], dtype=np.float32)
            mask = mask.transpose(2, 1, 0)
    else:
        array = loadmat(path, variable_names=[key])[key]
        mask = np.asarray(array[rows, cols, :], dtype=np.float32)

    return np.ascontiguousarray(mask)


def _load_measurement(
    path: Path,
    key: str,
    rows: slice,
    cols: slice,
) -> np.ndarray:
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as file:
            dataset = file[key]
            if dataset.ndim != 2:
                raise ValueError(f"{key!r} must be 2D, got {dataset.shape}.")

            measurement = np.asarray(dataset[cols, rows], dtype=np.float32).T
    else:
        array = loadmat(path, variable_names=[key])[key]
        measurement = np.asarray(array[rows, cols], dtype=np.float32)

    return np.ascontiguousarray(measurement)


def load_reconstruction_data(cfg: Config) -> ReconstructionData:
    mask_path = Path(cfg.mask_path)
    measurement_path = Path(cfg.measurement_path)
    for path in (mask_path, measurement_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    rows, cols, row_start0, col_start0 = centered_crop_slices(cfg)
    mask = _load_mask(mask_path, cfg.mask_key, rows, cols)
    measurement = _load_measurement(
        measurement_path,
        cfg.measurement_key,
        rows,
        cols,
    )
    measurement *= np.float32(cfg.measurement_gain)

    expected_shape = (cfg.crop_size, cfg.crop_size)
    if mask.shape[:2] != measurement.shape:
        raise ValueError(
            f"Spatial mismatch: mask={mask.shape[:2]}, measurement={measurement.shape}."
        )
    if measurement.shape != expected_shape:
        raise ValueError(
            f"Crop is outside the data: got {measurement.shape}, "
            f"expected {expected_shape}."
        )

    for name, array in (("mask", mask), ("measurement", measurement)):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf.")


    measurement_scale = max(
        float(np.percentile(np.abs(measurement), 99)),
        EPS,
    )


    mask_sum = mask.sum(axis=-1)
    valid = mask_sum > EPS
    if not np.any(valid):
        raise ValueError("Mask has no pixel with nonzero spectral throughput.")
    signal_scale = max(
        float(np.median(np.abs(measurement[valid] / mask_sum[valid]))),
        EPS,
    )

    return ReconstructionData(
        mask=mask,
        measurement=measurement,
        row_start0=row_start0,
        col_start0=col_start0,
        measurement_scale=measurement_scale,
        signal_scale=signal_scale,
    )


def describe(data: ReconstructionData) -> str:
    height, width, bands = data.shape
    return (
        f"HSI shape: {height} x {width} x {bands}\n"
        f"Measurement range: [{data.measurement.min():.6g}, "
        f"{data.measurement.max():.6g}]\n"
        f"Mask range: [{data.mask.min():.6g}, {data.mask.max():.6g}]\n"
        f"Optimization scales: measurement={data.measurement_scale:.6g}, "
        f"signal={data.signal_scale:.6g}\n"
        f"Crop (MATLAB 1-based): rows {data.row_start0 + 1}:"
        f"{data.row_start0 + height}, cols {data.col_start0 + 1}:"
        f"{data.col_start0 + width}"
    )


if __name__ == "__main__":
    print(describe(load_reconstruction_data(Config())))


def load_sequence(args, cfg: Config):
    rows, cols, r0, c0 = centered_crop_slices(cfg)
    t0 = time.perf_counter()
    mask = _load_mask(Path(args.mask), "dataCube", rows, cols)
    paths = sorted(Path(args.frame_dir).glob("*.mat"))
    if args.max_frames:
        paths = paths[: args.max_frames]
    frames = np.stack(
        [_load_measurement(p, "measurement", rows, cols) for p in paths], axis=0
    ).astype(np.float32)
    raw = frames.copy()
    frames *= np.float32(cfg.measurement_gain)
    print(f"Loaded mask {mask.shape} + {len(paths)} frames in {time.perf_counter()-t0:.1f}s; "
          f"crop rows {r0+1}:{r0+cfg.crop_size}, cols {c0+1}:{c0+cfg.crop_size} (1-based)", flush=True)
    return mask, frames, raw >= SATURATION, paths, (r0, c0)
