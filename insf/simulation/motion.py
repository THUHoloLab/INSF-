#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import time
import hashlib
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import nnls


METHODS = ("single_frame", "mean_static", "joint_static", "joint_true_warp")
DEFAULT_NS = (1, 2, 4, 8, 16, 32, 80)
DEFAULT_PHOTONS = (100.0, 1000.0, 10000.0)
DEFAULT_SPREADS = (0.0, 4.0, 8.0, 16.0)


def parse_csv_numbers(text: str, cast=float) -> tuple:
    return tuple(cast(x.strip()) for x in text.split(",") if x.strip())


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_real_mask(path: Path, crop_size: int, roi_start: tuple[int, int]) -> tuple[torch.Tensor, dict]:
    offset = (1000 - crop_size) // 2
    row_start = roi_start[0] - 1 + offset
    col_start = roi_start[1] - 1 + offset
    with h5py.File(path, "r") as f:
        d = f["dataCube"]
        raw = np.asarray(
            d[:, col_start : col_start + crop_size, row_start : row_start + crop_size],
            dtype=np.float32,
        )

    mask = torch.from_numpy(raw.transpose(2, 1, 0).copy())
    scale = float(mask.sum(dim=-1).median())
    mask = mask / max(scale, 1e-8)
    meta = {
        "source": str(path),
        "raw_shape_h5": list(raw.shape),
        "shape_hwl": list(mask.shape),
        "roi_start_matlab": list(roi_start),
        "crop_size": crop_size,
        "normalization": "divide by median spatial spectral throughput",
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "raw_mean": float(raw.mean()),
        "raw_mask_sum_median": scale,
    }
    return mask, meta


def make_spectral_basis(bands: int, k: int, seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed + 101)
    wavelength = torch.linspace(0.0, 1.0, bands, device=device)
    centers = torch.linspace(0.08, 0.92, k, device=device)
    widths = torch.linspace(0.055, 0.13, k, device=device)
    basis = torch.exp(-0.5 * ((wavelength[None, :] - centers[:, None]) / widths[:, None]) ** 2)
    basis = basis + 0.08 * torch.rand((k, bands), generator=g, device=device)
    basis = basis / basis.amax(dim=1, keepdim=True).clamp_min(1e-8)
    return basis


def make_spatial_basis(h: int, w: int, j: int, seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed + 211)
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device),
        torch.linspace(-1.0, 1.0, w, device=device),
        indexing="ij",
    )
    maps = [torch.ones((h, w), device=device)]

    maps.extend([
        0.5 + 0.5 * xx,
        0.5 + 0.5 * yy,
        0.5 + 0.5 * torch.sin(math.pi * xx) * torch.cos(math.pi * yy),
    ])

    for _ in range(max(0, j - len(maps))):
        cx, cy = (torch.rand(2, generator=g, device=device) * 1.4 - 0.7).tolist()
        sx, sy = (0.12 + 0.25 * torch.rand(2, generator=g, device=device)).tolist()
        blob = torch.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        maps.append(blob)
    out = torch.stack(maps[:j], dim=0)
    out = out / out.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return out


def generate_scene(
    mask: torch.Tensor,
    scene_id: int,
    spatial_rank: int,
    spectral_rank: int,
    device: torch.device,
) -> dict[str, torch.Tensor | float | int]:
    h, w, bands = mask.shape
    phi = make_spectral_basis(bands, spectral_rank, 1000 + scene_id, device)
    u = make_spatial_basis(h, w, spatial_rank, 2000 + scene_id, device)
    g = torch.Generator(device=device)
    g.manual_seed(3000 + scene_id)
    coeff = 0.15 + torch.rand((spatial_rank, spectral_rank), generator=g, device=device)

    coeff *= torch.linspace(0.7, 1.3, spectral_rank, device=device)[None, :]
    cube = torch.einsum("jhw,jk,kl->hwl", u, coeff, phi)
    cube_scale = cube.amax().clamp_min(1e-8)
    cube = cube / cube_scale
    return {
        "u": u,
        "phi": phi,


        "coeff": coeff / cube_scale,
        "cube": cube,
    }


def translation_grid(h: int, w: int, dx: float, dy: float, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device),
        torch.linspace(-1.0, 1.0, w, device=device),
        indexing="ij",
    )


    gx = xx + (2.0 * dx / max(w - 1, 1))
    gy = yy + (2.0 * dy / max(h - 1, 1))
    return torch.stack((gx, gy), dim=-1)[None]


def make_motion_offsets(total: int, spread: float, device: torch.device) -> torch.Tensor:
    t = torch.arange(total, device=device, dtype=torch.float32)
    phase = 2.0 * math.pi * t / max(total - 1, 1)
    dx = 0.5 * spread * torch.sin(phase)
    dy = 0.35 * spread * torch.sin(phase * 0.77)
    return torch.stack((dx, dy), dim=1)


def shifted_fields(
    u: torch.Tensor,
    a_nominal: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    t = offsets.shape[0]
    u_in = u[None].expand(t, -1, -1, -1)
    a_in = a_nominal.permute(2, 0, 1)[None].expand(t, -1, -1, -1)
    grids = torch.cat(
        [translation_grid(u.shape[1], u.shape[2], float(dx), float(dy), u.device) for dx, dy in offsets],
        dim=0,
    )
    u_shift = F.grid_sample(u_in, grids, mode="bilinear", padding_mode="border", align_corners=True)


    u_shift[0] = u
    if not bool(offsets.any()):
        u_shift = u_in
    return u_shift, a_in


def design_from_fields(u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:

    d = u[:, :, None] * a[:, None, :]
    return d.permute(0, 3, 4, 1, 2).reshape(d.shape[0], -1, d.shape[1] * d.shape[2])


def synthesize_observations(
    design: torch.Tensor,
    coeff: torch.Tensor,
    photons: float,
    seed: int,
    read_noise: float,
    clip_counts: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    clean = torch.einsum("tpd,d->tp", design, coeff.reshape(-1))
    clean = clean.clamp_min(0.0)
    g = torch.Generator(device=design.device)
    g.manual_seed(seed)
    noisy = torch.poisson(clean * photons, generator=g) / photons
    noisy = noisy + read_noise * torch.randn(noisy.shape, generator=g, device=design.device)
    clipped = 0.0
    if clip_counts > 0:
        clipped = float((noisy * photons >= clip_counts).float().mean().item())
        noisy = noisy.clamp_max(clip_counts / photons)
    return clean, noisy, clipped


def fit_ridge_from_normal_eq(
    design: torch.Tensor,
    observations: torch.Tensor,
    ridge: float,
    scale_ridge_by_frames: bool = False,
) -> torch.Tensor:
    n, p, d = design.shape


    design64 = design.to(torch.float64)
    obs64 = observations.to(torch.float64)
    gram = torch.einsum("tpd,tpe->de", design64, design64)
    rhs = torch.einsum("tpd,tp->d", design64, obs64)
    lam = ridge * (n if scale_ridge_by_frames else 1.0)
    eye = torch.eye(d, device=design.device, dtype=torch.float64)


    q = (gram + lam * eye).cpu().numpy()
    b = rhs.cpu().numpy()
    lower = np.linalg.cholesky(q)
    target = np.linalg.solve(lower, b)
    solution, _ = nnls(lower.T, target, maxiter=10 * d)
    return torch.as_tensor(solution, device=design.device, dtype=design.dtype)


def fit_static(
    d_nominal: torch.Tensor,
    y_train: torch.Tensor,
    ridge: float,
    joint: bool,
) -> torch.Tensor:
    if joint:
        d = d_nominal.expand(y_train.shape[0], -1, -1)
        return fit_ridge_from_normal_eq(d, y_train, ridge, scale_ridge_by_frames=True)
    return fit_ridge_from_normal_eq(d_nominal[None], y_train[:1], ridge)


def cube_from_coeff(
    u_nominal: torch.Tensor,
    phi: torch.Tensor,
    coeff: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("jhw,jk,kl->hwl", u_nominal, coeff.reshape(u_nominal.shape[0], -1), phi)


def cube_metrics(x_hat: torch.Tensor, x_true: torch.Tensor) -> tuple[float, float, float]:
    diff = x_hat - x_true
    mse = float(diff.square().mean().item())
    psnr = float(-10.0 * math.log10(max(mse, 1e-12)))
    dot = (x_hat * x_true).sum(dim=-1)
    denom = x_hat.norm(dim=-1) * x_true.norm(dim=-1)
    sam = torch.acos((dot / denom.clamp_min(1e-8)).clamp(-1.0, 1.0)).mean()
    return psnr, float(sam.item() * 180.0 / math.pi), mse


def measurement_rmse(
    design: torch.Tensor,
    coeff: torch.Tensor,
    y_target: torch.Tensor,
) -> float:
    pred = torch.einsum("tpd,d->tp", design, coeff.reshape(-1))
    return float(torch.sqrt((pred - y_target).square().mean()).item())


def sample_aligned_codes(mask: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    h, w, bands = mask.shape
    g = torch.Generator(device=mask.device).manual_seed(991)
    margin = float(offsets.abs().max()) + 2
    xy = torch.rand((128, 2), generator=g, device=mask.device)
    xy = margin + xy * torch.tensor([w - 1 - 2 * margin, h - 1 - 2 * margin], device=mask.device)
    grids = xy[None] - offsets[:, None, :]
    grids = 2 * grids / torch.tensor([w - 1, h - 1], device=mask.device) - 1
    out = F.grid_sample(mask.permute(2, 0, 1)[None].expand(len(offsets), -1, -1, -1),
                        grids[:, :, None, :], align_corners=True, padding_mode="border")
    return out.squeeze(-1).permute(2, 0, 1)


def effective_code_stats(codes: torch.Tensor, n: int) -> tuple[float, float]:
    singular = torch.linalg.svdvals(codes[:, :n].double())
    rank = (singular > singular[:, :1] * 1e-4).sum(dim=1)


    last = singular.gather(1, (rank.clamp_min(1) - 1)[:, None]).squeeze(1)
    cond = (singular[:, 0] / last.clamp_min(1e-12)).median()
    return float(rank.float().mean()), float(cond)


def run_condition(
    scene: dict,
    u_shift: torch.Tensor,
    a_shift: torch.Tensor,
    d_nominal: torch.Tensor,
    offsets: torch.Tensor,
    n_train: int,
    photons: float,
    noise_seed: int,
    ridge: float,
    read_noise: float,
    clip_counts: float,
    heldout: int,
    design_true: torch.Tensor,
    code_stats: tuple[float, float, float, float],
    observations: tuple[torch.Tensor, torch.Tensor, float],
) -> list[dict]:
    device = d_nominal.device


    clean, y_obs, clipped = observations
    y_train = y_obs[:n_train]
    y_mean = y_train.double().mean(dim=0, keepdim=True)
    coeff_single = fit_static(d_nominal, y_train, ridge, joint=False)
    coeff_mean = fit_ridge_from_normal_eq(d_nominal[None], y_mean, ridge)
    coeff_joint_static = fit_static(d_nominal, y_train, ridge, joint=True)
    d_train_true = design_true[:n_train]
    coeff_joint_true = fit_ridge_from_normal_eq(d_train_true, y_train, ridge, scale_ridge_by_frames=True)
    x_true = scene["cube"]
    x_single = cube_from_coeff(scene["u"], scene["phi"], coeff_single)
    x_mean = cube_from_coeff(scene["u"], scene["phi"], coeff_mean)
    x_static = cube_from_coeff(scene["u"], scene["phi"], coeff_joint_static)
    x_true_warp = cube_from_coeff(scene["u"], scene["phi"], coeff_joint_true)
    metric_map = {
        "single_frame": (coeff_single, x_single),
        "mean_static": (coeff_mean, x_mean),
        "joint_static": (coeff_joint_static, x_static),
        "joint_true_warp": (coeff_joint_true, x_true_warp),
    }

    val_design = design_true[-heldout:]
    val_y = y_obs[-heldout:]
    val_clean = clean[-heldout:]
    rank, cond, raw_rank, raw_cond = code_stats
    static_param_diff = float((coeff_mean - coeff_joint_static).norm().item() / coeff_mean.norm().clamp_min(1e-8).item())
    rows: list[dict] = []
    for method, (coeff_hat, x_hat) in metric_map.items():
        psnr, sam, mse = cube_metrics(x_hat, x_true)
        model_train = design_true[:n_train] if method == "joint_true_warp" else d_nominal[None].expand(n_train, -1, -1)
        model_val = val_design if method == "joint_true_warp" else d_nominal[None].expand(heldout, -1, -1)
        val_rmse = measurement_rmse(model_val, coeff_hat, val_y)
        train_rmse = measurement_rmse(model_train, coeff_hat, y_train)
        rows.append({
            "method": method,
            "n_train": n_train,
            "photon_level": photons,
            "noise_seed": noise_seed,
            "cube_psnr_db": psnr,
            "cube_sam_deg": sam,
            "cube_mse": mse,
            "train_measurement_rmse": train_rmse,
            "heldout_measurement_rmse": val_rmse,
            "heldout_clean_measurement_rmse": measurement_rmse(model_val, coeff_hat, val_clean),
            "heldout_oracle_warp_rmse": measurement_rmse(val_design, coeff_hat, val_y),
            "heldout_temporal_mean_rmse": float(((y_mean - val_y).square().mean()).sqrt()),
            "heldout_noise_rmse": float(((val_clean - val_y).square().mean()).sqrt()),
            "clip_fraction": clipped,
            "code_rank_mean": rank,
            "code_condition_median": cond,
            "raw_601_code_rank_mean": raw_rank,
            "raw_601_nonzero_condition_median": raw_cond,
            "mean_vs_joint_static_param_rel_diff": static_param_diff,
            "max_dx_px": float(offsets[:n_train, 0].abs().max().item()),
            "max_dy_px": float(offsets[:n_train, 1].abs().max().item()),
        })
    return rows


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--roi-start", type=int, nargs=2, default=(200, 200))
    parser.add_argument("--num-scenes", type=int, default=5)
    parser.add_argument("--num-noise-seeds", type=int, default=5)
    parser.add_argument("--n-values", default=','.join(map(str, DEFAULT_NS)))
    parser.add_argument("--photon-levels", default=','.join(map(str, DEFAULT_PHOTONS)))
    parser.add_argument("--motion-spreads", default=','.join(map(str, DEFAULT_SPREADS)))
    parser.add_argument("--heldout", type=int, default=16)
    parser.add_argument("--spatial-rank", type=int, default=6)
    parser.add_argument("--spectral-rank", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--read-noise", type=float, default=0.002)
    parser.add_argument("--clip-counts", type=float, default=0.0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this simulation")
    device = torch.device("cuda")
    set_seed(20261007 + args.shard)
    mask, mask_meta = load_real_mask(args.mask, args.crop_size, tuple(args.roi_start))
    mask = mask.to(device)
    h, w, bands = mask.shape
    n_values = parse_csv_numbers(args.n_values, int)
    photons = parse_csv_numbers(args.photon_levels, float)
    spreads = parse_csv_numbers(args.motion_spreads, float)
    if args.smoke:
        n_values, photons, spreads = (1, 4, 8), (1000.0,), (0.0, 8.0)
        args.num_scenes, args.num_noise_seeds, args.heldout = 1, 1, 4
    max_n = max(n_values)
    total_frames = max_n + args.heldout
    scene_ids = [i for i in range(args.num_scenes) if i % args.num_shards == args.shard]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "host": os.uname().nodename,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "mask": mask_meta,
        "simulation": {
            "scene_ids": scene_ids,
            "num_scenes_total": args.num_scenes,
            "num_noise_seeds": args.num_noise_seeds,
            "n_values": list(n_values),
            "photon_levels": list(photons),
            "motion_spreads_px": list(spreads),
            "heldout_frames": args.heldout,
            "spatial_rank": args.spatial_rank,
            "spectral_rank": args.spectral_rank,
            "ridge": args.ridge,
            "read_noise": args.read_noise,
            "clip_counts": args.clip_counts,
            "methods": list(METHODS),
            "interpretation": "controlled low-rank field simulation with real mask; not real-camera image-domain validation",
        },
    }
    (args.output.with_suffix(".meta.json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    all_rows: list[dict] = []
    for scene_id in scene_ids:
        print(f"scene={scene_id} start", flush=True)
        scene = generate_scene(mask, scene_id, args.spatial_rank, args.spectral_rank, device)
        a_nom = torch.einsum("hwl,kl->hwk", mask, scene["phi"])

        u_nom = scene["u"][None]
        a_nom_cf = a_nom.permute(2, 0, 1)[None]
        d_nom = design_from_fields(u_nom, a_nom_cf)[0]
        for spread in spreads:
            offsets = make_motion_offsets(total_frames, float(spread), device)
            u_shift, a_shift = shifted_fields(scene["u"], a_nom, offsets)
            design_true = design_from_fields(u_shift, a_shift)
            aligned_codes = sample_aligned_codes(mask, offsets)
            raw_rank = 1.0
            raw_cond = 1.0
            for photon in photons:
                for noise_seed in range(args.num_noise_seeds):
                    observations = synthesize_observations(
                        design_true,
                        scene["coeff"],
                        float(photon),
                        int(noise_seed),
                        args.read_noise,
                        args.clip_counts,
                    )
                    for n_train in n_values:
                        rank, cond = effective_code_stats(aligned_codes, int(n_train))
                        rows = run_condition(
                            scene,
                            u_shift,
                            a_shift,
                            d_nom,
                            offsets,
                            int(n_train),
                            float(photon),
                            int(noise_seed),
                            args.ridge,
                            args.read_noise,
                            args.clip_counts,
                            args.heldout,
                            design_true,
                            (rank, cond, raw_rank, raw_cond),
                            observations,
                        )
                        for row in rows:
                            row.update({"scene_id": scene_id, "motion_spread_px": spread})
                        all_rows.extend(rows)
            del u_shift, a_shift
            torch.cuda.empty_cache()
        print(f"scene={scene_id} rows={len(all_rows)}", flush=True)
    write_csv(args.output, all_rows)
    static_rows = [
        r for r in all_rows
        if float(r["motion_spread_px"]) == 0.0
        and r["method"] in ("mean_static", "joint_static")
    ]
    summary = {
        "rows": len(all_rows),
        "scene_ids": scene_ids,
        "output": str(args.output),
        "static_equivalence_max_rel_diff": max(
            (float(r["mean_vs_joint_static_param_rel_diff"]) for r in static_rows),
            default=float("nan"),
        ),
        "static_equivalence_scope": "motion_spread_px == 0; mean_static vs joint_static parameter relative difference",
    }
    (args.output.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
