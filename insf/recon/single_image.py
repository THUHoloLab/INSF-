import sys
from pathlib import Path
from insf import paths
import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import savemat

from insf.config import Config
from insf.data.io import ReconstructionData, describe, load_reconstruction_data
from insf.models.field import SpatialSpectralField, count_parameters


MAT_V5_LIMIT_BYTES = 1_800_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct a hyperspectral cube with the FiLM spatial-spectral INR."
    )
    parser.add_argument("--mask", type=Path, help="Override the mask path of the configuration.")
    parser.add_argument(
        "--measurement",
        type=Path,
        help="Override the measurement path of the configuration.",
    )
    parser.add_argument("--crop-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume", type=Path, help="Resume training from checkpoint.pt.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run two minibatches only and save nothing.",
    )
    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> None:
    if args.mask is not None:
        cfg.mask_path = args.mask.resolve()
    if args.measurement is not None:
        cfg.measurement_path = args.measurement.resolve()
    if args.crop_size is not None:
        cfg.crop_size = args.crop_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    if args.smoke_test:
        cfg.crop_size = 16
        cfg.epochs = 1
        cfg.batch_size = 128
        cfg.steps_per_epoch = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_xy_grid(
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    ys = (torch.arange(height, device=device) + 0.5) / height
    xs = (torch.arange(width, device=device) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2).float().contiguous()


def charbonnier_loss(values: torch.Tensor, epsilon: float) -> torch.Tensor:
    return (torch.sqrt(values.square() + epsilon**2) - epsilon).mean()


def cosine_learning_rate(cfg: Config, epoch: int) -> float:
    progress = epoch / max(cfg.epochs - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_learning_rate + cosine * (cfg.learning_rate - cfg.min_learning_rate)


def spatial_tv_loss(
    model: SpatialSpectralField,
    full_xy: torch.Tensor,
    wavelengths: torch.Tensor,
    data: ReconstructionData,
    cfg: Config,
) -> torch.Tensor:
    wavelength = wavelengths[
        torch.randint(wavelengths.numel(), (1,), device=full_xy.device)
    ]
    height, width, _ = data.shape
    image = model(full_xy, wavelength).reshape(height, width)
    horizontal = image[:, 1:] - image[:, :-1]
    vertical = image[1:, :] - image[:-1, :]
    return 0.5 * (
        charbonnier_loss(
            horizontal / data.signal_scale,
            cfg.tv_epsilon,
        )
        + charbonnier_loss(
            vertical / data.signal_scale,
            cfg.tv_epsilon,
        )
    )


def compute_losses(
    model: SpatialSpectralField,
    batch_xy: torch.Tensor,
    full_xy: torch.Tensor,
    wavelengths: torch.Tensor,
    mask: torch.Tensor,
    measurement: torch.Tensor,
    data: ReconstructionData,
    cfg: Config,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    spectrum = model(batch_xy, wavelengths)


    estimated_measurement = (spectrum * mask).sum(dim=-1)
    data_loss = F.smooth_l1_loss(
        estimated_measurement / data.measurement_scale,
        measurement / data.measurement_scale,
        beta=cfg.huber_beta,
    )
    spectral_tv = charbonnier_loss(
        torch.diff(spectrum, dim=-1) / data.signal_scale,
        cfg.tv_epsilon,
    )
    spatial_tv = spatial_tv_loss(model, full_xy, wavelengths, data, cfg)

    total = (
        data_loss
        + cfg.spectral_tv_weight * spectral_tv
        + cfg.spatial_tv_weight * spatial_tv
    )
    return total, {
        "data": data_loss,
        "spectral_tv": spectral_tv,
        "spatial_tv": spatial_tv,
    }


def save_checkpoint(
    path: Path,
    model: SpatialSpectralField,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: Config,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.to_dict(),
        },
        path,
    )


@torch.inference_mode()
def render_band(
    model: SpatialSpectralField,
    xy: torch.Tensor,
    wavelength: torch.Tensor,
    height: int,
    width: int,
    pixel_batch: int,
) -> np.ndarray:
    was_training = model.training
    model.eval()
    output = np.empty(height * width, dtype=np.float32)

    for start in range(0, xy.shape[0], pixel_batch):
        stop = min(start + pixel_batch, xy.shape[0])
        output[start:stop] = model(xy[start:stop], wavelength)[:, 0].cpu().numpy()

    model.train(was_training)
    return output.reshape(height, width)


def save_preview(image: np.ndarray, path: Path) -> None:
    low, high = np.percentile(image, [1, 99])
    normalized = np.clip((image - low) / max(high - low, 1e-8), 0.0, 1.0)
    plt.imsave(path, normalized, cmap="gray", vmin=0.0, vmax=1.0)


def save_epoch_artifacts(
    run_dir: Path,
    model: SpatialSpectralField,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    xy: torch.Tensor,
    wavelengths: torch.Tensor,
    data: ReconstructionData,
    cfg: Config,
) -> None:
    save_checkpoint(run_dir / "checkpoint.pt", model, optimizer, epoch, cfg)
    band_index = min(max(cfg.preview_band - 1, 0), wavelengths.numel() - 1)
    preview = render_band(
        model,
        xy,
        wavelengths[band_index : band_index + 1],
        data.shape[0],
        data.shape[1],
        cfg.reconstruction_pixel_batch,
    )
    save_preview(preview, run_dir / f"preview_epoch_{epoch:03d}.png")


def train(
    model: SpatialSpectralField,
    optimizer: torch.optim.Optimizer,
    xy: torch.Tensor,
    wavelengths: torch.Tensor,
    mask: torch.Tensor,
    measurement: torch.Tensor,
    data: ReconstructionData,
    cfg: Config,
    run_dir: Path | None,
    first_epoch: int = 0,
) -> None:
    num_pixels = xy.shape[0]
    max_steps = num_pixels // cfg.batch_size
    requested_steps = max_steps if cfg.steps_per_epoch is None else cfg.steps_per_epoch
    steps = min(requested_steps, max_steps)
    if steps < 1:
        raise ValueError(
            f"batch_size={cfg.batch_size} is too large for {num_pixels} pixels."
        )

    for epoch in range(first_epoch, cfg.epochs):
        model.train()
        learning_rate = cosine_learning_rate(cfg, epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        permutation = torch.randperm(num_pixels, device=xy.device)
        sums = {
            "total": 0.0,
            "data": 0.0,
            "spectral_tv": 0.0,
            "spatial_tv": 0.0,
        }

        for step in range(steps):
            start = step * cfg.batch_size
            indices = permutation[start : start + cfg.batch_size]
            optimizer.zero_grad(set_to_none=True)

            loss, terms = compute_losses(
                model,
                xy[indices],
                xy,
                wavelengths,
                mask[indices],
                measurement[indices],
                data,
                cfg,
            )
            loss.backward()
            optimizer.step()

            sums["total"] += loss.detach().item()
            for name, value in terms.items():
                sums[name] += value.detach().item()

        means = {name: value / steps for name, value in sums.items()}
        print(
            f"Epoch {epoch + 1:03d}/{cfg.epochs}: "
            f"lr={learning_rate:.3g}, "
            f"total={means['total']:.6g}, data={means['data']:.6g}, "
            f"spectral_tv={means['spectral_tv']:.6g}, "
            f"spatial_tv={means['spatial_tv']:.6g}"
        )

        should_save = (epoch + 1) % cfg.save_every == 0 or epoch + 1 == cfg.epochs
        if run_dir is not None and should_save:
            save_epoch_artifacts(
                run_dir,
                model,
                optimizer,
                epoch + 1,
                xy,
                wavelengths,
                data,
                cfg,
            )


@torch.inference_mode()
def reconstruct_cube(
    model: SpatialSpectralField,
    xy: torch.Tensor,
    wavelengths: torch.Tensor,
    height: int,
    width: int,
    pixel_batch: int,
    spectral_batch: int,
) -> np.ndarray:
    model.eval()
    num_bands = wavelengths.numel()
    flat = np.empty((height * width, num_bands), dtype=np.float32)
    print("Reconstructing full HSI...")

    for pixel_start in range(0, xy.shape[0], pixel_batch):
        pixel_stop = min(pixel_start + pixel_batch, xy.shape[0])
        for band_start in range(0, num_bands, spectral_batch):
            band_stop = min(band_start + spectral_batch, num_bands)
            flat[pixel_start:pixel_stop, band_start:band_stop] = (
                model(
                    xy[pixel_start:pixel_stop],
                    wavelengths[band_start:band_stop],
                )
                .cpu()
                .numpy()
            )

    return flat.reshape(height, width, num_bands)


def save_result(
    run_dir: Path,
    cube: np.ndarray,
    data: ReconstructionData,
    cfg: Config,
) -> Path:
    if cube.nbytes >= MAT_V5_LIMIT_BYTES:
        raise ValueError(
            "The HSI is too large for scipy savemat; use chunked HDF5 instead."
        )

    output_path = run_dir / "reconstruction_result.mat"
    savemat(
        output_path,
        {
            "hsi_pred": cube,
            "crop_size": np.int32(cfg.crop_size),
            "start_x": np.int32(data.row_start0 + 1),
            "start_y": np.int32(data.col_start0 + 1),
            "end_x": np.int32(data.row_start0 + data.shape[0]),
            "end_y": np.int32(data.col_start0 + data.shape[1]),
        },
        do_compression=cfg.compress_mat,
    )
    return output_path


def create_run_directory(cfg: Config) -> Path:
    run_dir = Path(cfg.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir


def load_training_state(
    path: Path,
    model: SpatialSpectralField,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["epoch"])


def main() -> None:
    args = parse_args()
    cfg = Config()
    apply_overrides(cfg, args)

    if not torch.cuda.is_available():
        raise RuntimeError("tiny-cuda-nn requires a CUDA GPU.")
    device = torch.device("cuda")
    set_seed(cfg.seed)


    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False

    data = load_reconstruction_data(cfg)
    print(describe(data))
    height, width, num_bands = data.shape

    xy = make_xy_grid(height, width, device)
    wavelengths = torch.linspace(0.0, 1.0, num_bands, device=device)
    mask = torch.from_numpy(data.mask.reshape(-1, num_bands)).to(device)
    measurement = torch.from_numpy(data.measurement.reshape(-1)).to(device)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    first_epoch = 0
    if args.resume is not None:
        first_epoch = load_training_state(args.resume, model, optimizer, device)
        print(f"Resumed from epoch {first_epoch}: {args.resume}")

    run_dir = None if args.smoke_test else create_run_directory(cfg)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    if run_dir is not None:
        print(f"Run directory: {run_dir}")

    train(
        model,
        optimizer,
        xy,
        wavelengths,
        mask,
        measurement,
        data,
        cfg,
        run_dir,
        first_epoch,
    )

    if args.smoke_test:
        print("Smoke test passed.")
        return

    cube = reconstruct_cube(
        model,
        xy,
        wavelengths,
        height,
        width,
        cfg.reconstruction_pixel_batch,
        cfg.reconstruction_spectral_batch,
    )
    output_path = save_result(run_dir, cube, data, cfg)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
