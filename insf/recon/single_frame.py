from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from insf.config import Config
from insf.data.io import _load_mask, _load_measurement, centered_crop_slices, load_sequence
from insf.models.field import SpatialSpectralField, SpatioTemporalField, count_parameters, tcnn
from insf.forward import (SATURATION, set_seed, sync_time, charbonnier, cosine_lr, make_xy, frame_scales,
                          render_cube)
from insf.evaluation.color import WAVELENGTHS_NM, cie_matrix

PREVIEW_BANDS_NM = (450, 550, 650, 750, 850)


def fit_frame(model, opt, xy, wl, mask, y, meas_scale, sig_scale, h, w, cfg,
              epochs, lr0, lr_min, log):
    n = xy.shape[0]
    steps = n // cfg.batch_size
    total = epochs * steps
    k = 0
    for ep in range(epochs):
        perm = torch.randperm(n, device=xy.device)
        acc = torch.zeros(3, device=xy.device)
        for s in range(steps):
            lr = cosine_lr(k, total, lr0, lr_min)
            for g in opt.param_groups:
                g["lr"] = lr
            idx = perm[s * cfg.batch_size:(s + 1) * cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            spec = model(xy[idx], wl)
            est = (spec * mask[idx]).sum(-1)
            data = F.smooth_l1_loss(est / meas_scale, y[idx] / meas_scale, beta=cfg.huber_beta)
            stv = charbonnier(torch.diff(spec, dim=-1) / sig_scale, cfg.tv_epsilon)
            band = wl[torch.randint(wl.numel(), (1,), device=xy.device)]
            img = model(xy, band).reshape(h, w)
            ptv = 0.5 * (charbonnier((img[:, 1:] - img[:, :-1]) / sig_scale, cfg.tv_epsilon)
                         + charbonnier((img[1:] - img[:-1]) / sig_scale, cfg.tv_epsilon))
            loss = data + cfg.spectral_tv_weight * stv + cfg.spatial_tv_weight * ptv
            loss.backward()
            opt.step()
            acc += torch.stack([data.detach(), stv.detach(), ptv.detach()])
            k += 1
        acc = (acc / steps).tolist()
        log(f"  epoch {ep+1}/{epochs} lr={lr:.2e} data={acc[0]:.5g} spec_tv={acc[1]:.5g} spat_tv={acc[2]:.5g}")


class Recorder:

    def __init__(self, args, out_dir: Path, mask, frames, sat, paths, rgb_mat, h, w):
        self.args, self.out, self.h, self.w = args, out_dir, h, w
        self.mask, self.frames, self.sat, self.paths = mask, frames, sat, paths
        self.rgb_mat = torch.from_numpy(rgb_mat).to(mask.device)
        self.band_idx = [int(nm - WAVELENGTHS_NM[0]) for nm in PREVIEW_BANDS_NM]
        for sub in ("rgb_linear", "bands", "pred_measurement"):
            (out_dir / sub).mkdir(parents=True, exist_ok=True)
        self.cube_dir = Path(args.cube_dir) if args.cube_dir else None
        if self.cube_dir:
            self.cube_dir.mkdir(parents=True, exist_ok=True)
        self.prev_cube = None
        self.prev_pred = None
        self.rows: list[dict] = []

    def record(self, t: int, cube: torch.Tensor, gain=1.0, bias=0.0, split="train",
               fit_seconds=float("nan"), extra=None) -> dict:
        y = self.frames[t]
        base = (cube * self.mask).sum(-1)
        pred = gain * base + bias
        res = pred - y
        unsat = ~self.sat[t]
        row = {
            "frame": t,
            "file": self.paths[t].name,
            "split": split,
            "rmse": float(res.square().mean().sqrt()),
            "rel_rmse": float(res.square().mean().sqrt() / y.std()),
            "rmse_unsaturated": float(res[unsat].square().mean().sqrt()),
            "mae": float(res.abs().mean()),
            "gain": float(gain),
            "bias": float(bias),
            "cube_mean": float(cube.mean()),
            "fit_seconds": fit_seconds,
        }

        if self.prev_cube is not None:
            row["flicker_cube"] = float((cube - self.prev_cube).abs().mean() / cube.abs().mean())
            row["flicker_pred_meas"] = float((pred - self.prev_pred).abs().mean() / pred.abs().mean())
            ym = self.frames[t - 1]
            row["flicker_meas"] = float((y - ym).abs().mean() / y.abs().mean())
        else:
            row["flicker_cube"] = row["flicker_pred_meas"] = row["flicker_meas"] = float("nan")
        if extra:
            row.update(extra)
        self.prev_cube, self.prev_pred = cube, pred

        h, w = self.h, self.w
        rgb = (cube @ self.rgb_mat).reshape(h, w, 3).cpu().numpy()
        np.save(self.out / "rgb_linear" / f"frame_{t:03d}.npy", rgb.astype(np.float32))
        bands = cube[:, self.band_idx].reshape(h, w, -1).cpu().numpy()
        np.save(self.out / "bands" / f"frame_{t:03d}.npy", bands.astype(np.float32))
        np.save(self.out / "pred_measurement" / f"frame_{t:03d}.npy",
                pred.reshape(h, w).cpu().numpy().astype(np.float32))
        if self.cube_dir:
            np.save(self.cube_dir / f"frame_{t:03d}.npy",
                    cube.reshape(h, w, -1).cpu().numpy().astype(np.float16))
        self.rows.append(row)
        print(f"[frame {t:03d} {split}] rmse={row['rmse']:.3f} rel={row['rel_rmse']:.4f} "
              f"flicker cube={row['flicker_cube']:.4f} meas={row['flicker_meas']:.4f} "
              f"fit={fit_seconds:.1f}s", flush=True)
        self.flush()
        return row

    def flush(self):
        keys = []
        for r in self.rows:
            keys += [k for k in r if k not in keys]
        with open(self.out / "metrics_per_frame.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(self.rows)


def summarize(rows, extra) -> dict:
    def mean(key, split=None):
        vals = [r[key] for r in rows if (split is None or r["split"] == split)
                and r.get(key) is not None and not math.isnan(r[key])]
        return float(np.mean(vals)) if vals else None

    out = {
        "frames": len(rows),
        "mean_rmse_all": mean("rmse"),
        "mean_rmse_train": mean("rmse", "train"),
        "mean_rmse_holdout": mean("rmse", "holdout"),
        "mean_rel_rmse_train": mean("rel_rmse", "train"),
        "mean_rel_rmse_holdout": mean("rel_rmse", "holdout"),
        "mean_rmse_unsat_train": mean("rmse_unsaturated", "train"),
        "mean_flicker_cube": mean("flicker_cube"),
        "mean_flicker_pred_meas": mean("flicker_pred_meas"),
        "mean_flicker_meas": mean("flicker_meas"),
    }
    out.update(extra)
    return out


def build_2d(cfg, sig_scale, device):
    return SpatialSpectralField(
        num_spectral_knots=cfg.num_spectral_knots, output_scale=sig_scale,
        n_levels=cfg.n_levels, n_features_per_level=cfg.n_features_per_level,
        log2_hashmap_size=cfg.log2_hashmap_size, base_resolution=cfg.base_resolution,
        per_level_scale=cfg.per_level_scale, hidden_dim=cfg.hidden_dim,
        hidden_layers=cfg.hidden_layers).to(device)


def run_per_frame(args, cfg, rec, xy, wl, mask_t, frames_t, mask_sum, h, w, log, warm: bool):
    model = opt = None
    t_start = sync_time()
    epochs_used = []
    for t in range(frames_t.shape[0]):
        y = frames_t[t]
        meas_scale, sig_scale = frame_scales(rec.frames_np[t], mask_sum)
        t0 = sync_time()
        if not warm or model is None:
            set_seed(cfg.seed)
            model = build_2d(cfg, sig_scale, xy.device)
            opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
            epochs, lr0 = args.epochs, cfg.learning_rate
        else:
            epochs, lr0 = args.warm_epochs, args.warm_lr
        log(f"frame {t} ({rec.paths[t].name}): {epochs} epochs, lr0={lr0:g}")
        fit_frame(model, opt, xy, wl, mask_t, y, meas_scale, sig_scale, h, w, cfg,
                  epochs, lr0, cfg.min_learning_rate, log)
        fit_s = sync_time() - t0
        epochs_used.append(epochs)
        cube = render_cube(model, xy, wl)
        rec.record(t, cube, fit_seconds=fit_s, extra={"epochs": epochs})
    total = sync_time() - t_start
    return {"total_wall_seconds": total, "wall_seconds_per_frame": total / frames_t.shape[0],
            "epochs_per_frame": epochs_used, "total_epochs": int(sum(epochs_used)),
            "parameters": count_parameters(model)}


def run_e3(args, cfg, rec, xy, wl, mask_t, frames_t, mask_sum, h, w, log):
    T = frames_t.shape[0]
    device = xy.device
    holdout = np.zeros(T, bool)
    if args.holdout_every > 0:
        holdout[np.arange(T) % args.holdout_every == args.holdout_every - 1] = True
    train_ids = np.flatnonzero(~holdout)
    train_t = torch.from_numpy(train_ids).to(device)
    log(f"E3: {len(train_ids)} training frames, {int(holdout.sum())} held-out: {np.flatnonzero(holdout).tolist()}")


    finest = cfg.base_resolution * cfg.per_level_scale ** (cfg.n_levels - 1)
    extent = min(1.0, T / (args.frames_per_cell * finest))
    taus = torch.tensor(0.5 - extent / 2 + (np.arange(T) + 0.5) / T * extent,
                        dtype=torch.float32, device=device)
    log(f"tau extent={extent:.4f} (finest level res {finest:.0f})")

    train_frames_np = rec.frames_np[train_ids]
    meas_scale = max(float(np.percentile(np.abs(train_frames_np), 99)), 1e-6)
    ref = np.median(train_frames_np, axis=0)
    _, sig_scale = frame_scales(ref, mask_sum)

    set_seed(cfg.seed)
    model = SpatioTemporalField(
        log2_hashmap_size_3d=args.log2_hashmap_3d,
        num_spectral_knots=cfg.num_spectral_knots, output_scale=sig_scale,
        n_levels=cfg.n_levels, n_features_per_level=cfg.n_features_per_level,
        log2_hashmap_size=cfg.log2_hashmap_size, base_resolution=cfg.base_resolution,
        per_level_scale=cfg.per_level_scale, hidden_dim=cfg.hidden_dim,
        hidden_layers=cfg.hidden_layers).to(device)
    log_gain = torch.nn.Parameter(torch.zeros(T, device=device))
    bias = torch.nn.Parameter(torch.zeros(T, device=device))
    opt = torch.optim.Adam(list(model.parameters()) + [log_gain, bias], lr=cfg.learning_rate)
    log(f"E3 parameters: {count_parameters(model):,} + {2*T} frame parameters")

    n_pix = xy.shape[0]
    B = args.e3_batch
    total = args.e3_steps
    t_start = sync_time()
    acc = torch.zeros(3, device=device)
    for k in range(total):
        lr = cosine_lr(k, total, cfg.learning_rate, cfg.min_learning_rate)
        for g in opt.param_groups:
            g["lr"] = lr
        if args.center_gain:

            lg_all = log_gain - log_gain[train_t].mean()
            b_all = bias - bias[train_t].mean()
        else:
            lg_all, b_all = log_gain, bias
        f = train_t[torch.randint(len(train_ids), (B,), device=device)]
        p = torch.randint(n_pix, (B,), device=device)
        coords = torch.cat([xy[p], taus[f, None]], dim=1)
        opt.zero_grad(set_to_none=True)
        spec = model(coords, wl)
        base = (spec * mask_t[p]).sum(-1)
        pred = lg_all[f].exp() * base + b_all[f]
        data = F.smooth_l1_loss(pred / meas_scale, frames_t[f, p] / meas_scale, beta=cfg.huber_beta)
        stv = charbonnier(torch.diff(spec, dim=-1) / sig_scale, cfg.tv_epsilon)
        loss = data + cfg.spectral_tv_weight * stv
        if args.spatial_tv_every and k % args.spatial_tv_every == 0:
            ft = train_t[torch.randint(len(train_ids), (1,), device=device)]
            band = wl[torch.randint(wl.numel(), (1,), device=device)]
            full = torch.cat([xy, taus[ft].expand(n_pix, 1)], dim=1)
            img = model(full, band).reshape(h, w)
            ptv = 0.5 * (charbonnier((img[:, 1:] - img[:, :-1]) / sig_scale, cfg.tv_epsilon)
                         + charbonnier((img[1:] - img[:-1]) / sig_scale, cfg.tv_epsilon))
            loss = loss + cfg.spatial_tv_weight * ptv
            acc[2] += ptv.detach()
        lg = lg_all[train_t]
        loss = (loss + args.gain_prior * lg.square().mean()
                + args.bias_prior * (b_all[train_t] / meas_scale).square().mean()
                + args.gain_smooth * torch.diff(lg).square().mean())
        loss.backward()
        opt.step()
        acc[0] += data.detach()
        acc[1] += stv.detach()
        if (k + 1) % args.log_every == 0:
            a = acc.tolist()
            n_tv = max(args.log_every // max(args.spatial_tv_every, 1), 1)
            log(f"  step {k+1}/{total} lr={lr:.2e} data={a[0]/args.log_every:.5g} "
                f"spec_tv={a[1]/args.log_every:.5g} spat_tv={a[2]/n_tv:.5g} "
                f"elapsed={time.perf_counter()-t_start:.0f}s")
            acc.zero_()
    train_s = sync_time() - t_start


    with torch.no_grad():
        g_t, b_t = log_gain.detach().clone(), bias.detach().clone()
        if args.center_gain:
            g_t -= g_t[train_t].mean()
            b_t -= b_t[train_t].mean()
    g_np = g_t.cpu().numpy()
    b_np = b_t.cpu().numpy()
    g_use = g_np.copy()
    b_use = b_np.copy()
    if holdout.any():
        g_use[holdout] = np.interp(np.flatnonzero(holdout), train_ids, g_np[train_ids])
        b_use[holdout] = np.interp(np.flatnonzero(holdout), train_ids, b_np[train_ids])
    render_s = []
    for t in range(T):
        t0 = sync_time()
        coords = torch.cat([xy, taus[t].expand(n_pix, 1)], dim=1)
        cube = render_cube(model, coords, wl)
        extra = {}
        if holdout[t]:

            base = (cube * mask_t).sum(-1)
            A = torch.stack([base, torch.ones_like(base)], 1).double()
            sol = torch.linalg.lstsq(A, frames_t[t].double()[:, None]).solution.squeeze(1)
            res = (A @ sol).float() - frames_t[t]
            extra["rmse_affine_fit"] = float(res.square().mean().sqrt())
            prev_y, next_y = frames_t[t - 1], frames_t[min(t + 1, T - 1)]
            extra["rmse_copy_prev_meas"] = float((prev_y - frames_t[t]).square().mean().sqrt())
            extra["rmse_avg_neighbor_meas"] = float((0.5 * (prev_y + next_y) - frames_t[t]).square().mean().sqrt())
        rec.record(t, cube, gain=float(np.exp(g_use[t])), bias=float(b_use[t]),
                   split="holdout" if holdout[t] else "train",
                   fit_seconds=train_s / T, extra=extra)
        render_s.append(sync_time() - t0)
    total_s = train_s + sum(render_s)
    torch.save({"model": model.state_dict(), "log_gain": g_np, "bias": b_np,
                "taus": taus.cpu(), "holdout": holdout}, rec.out / "e3_checkpoint.pt")
    hold_rows = [r for r in rec.rows if r["split"] == "holdout"]
    extra = {}
    for key in ("rmse_affine_fit", "rmse_copy_prev_meas", "rmse_avg_neighbor_meas"):
        if hold_rows:
            extra[f"holdout_mean_{key}"] = float(np.mean([r[key] for r in hold_rows]))
    extra.update({"train_seconds": train_s, "total_wall_seconds": total_s,
                  "wall_seconds_per_frame": total_s / T, "steps": total, "batch": B,
                  "epoch_equivalents": total * B / (len(train_ids) * n_pix),
                  "tau_extent": extent, "holdout_frames": np.flatnonzero(holdout).tolist(),
                  "parameters": count_parameters(model) + 2 * T})
    return extra


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["e1", "e2", "e3"], required=True)
    p.add_argument("--frame-dir", type=Path, required=True)
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--xyzbar", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cube-dir", type=Path, help="save float16 [H,W,601] cubes here")
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--epochs", type=int, default=40, help="E1 epochs / E2 first-frame epochs")
    p.add_argument("--warm-epochs", type=int, default=5)
    p.add_argument("--warm-lr", type=float, default=3e-4)
    p.add_argument("--e3-steps", type=int, default=20000)
    p.add_argument("--e3-batch", type=int, default=4096)
    p.add_argument("--spatial-tv-every", type=int, default=4)
    p.add_argument("--holdout-every", type=int, default=5)
    p.add_argument("--frames-per-cell", type=float, default=1.5)
    p.add_argument("--log2-hashmap-3d", type=int, default=22)
    p.add_argument("--gain-prior", type=float, default=1e-3)
    p.add_argument("--bias-prior", type=float, default=1e-4)
    p.add_argument("--gain-smooth", type=float, default=1e-3)
    p.add_argument("--no-center-gain", dest="center_gain", action="store_false",
                   help="let the mean per-frame gain/offset float (v1 behaviour)")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--quiet-epochs", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    cfg.crop_size = args.crop_size
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)

    mask_np, frames_np, sat_np, paths, (r0, c0) = load_sequence(args, cfg)
    h, w, L = mask_np.shape
    mask_sum = mask_np.sum(-1)
    xy = make_xy(h, w, device)
    wl = torch.linspace(0.0, 1.0, L, device=device)
    mask_t = torch.from_numpy(mask_np.reshape(-1, L)).to(device)
    frames_t = torch.from_numpy(frames_np.reshape(len(paths), -1)).to(device)
    sat_t = torch.from_numpy(sat_np.reshape(len(paths), -1)).to(device)

    rec = Recorder(args, args.out, mask_t, frames_t, sat_t, paths, cie_matrix(args.xyzbar), h, w)
    rec.frames_np = frames_np
    log_file = open(args.out / "train_log.txt", "a")

    def log(msg):
        log_file.write(msg + "\n")
        log_file.flush()
        if not (args.quiet_epochs and msg.startswith("  epoch")):
            print(msg, flush=True)

    meta = {"args": {k: str(v) for k, v in vars(args).items()}, "config": cfg.to_dict(),
            "crop_rows_1based": [r0 + 1, r0 + h], "crop_cols_1based": [c0 + 1, c0 + w],
            "frames": [p.name for p in paths], "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__}
    (args.out / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    if args.method == "e1":
        extra = run_per_frame(args, cfg, rec, xy, wl, mask_t, frames_t, mask_sum, h, w, log, warm=False)
    elif args.method == "e2":
        extra = run_per_frame(args, cfg, rec, xy, wl, mask_t, frames_t, mask_sum, h, w, log, warm=True)
    else:
        extra = run_e3(args, cfg, rec, xy, wl, mask_t, frames_t, mask_sum, h, w, log)
    summary = summarize(rec.rows, extra)
    summary["method"] = args.method
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
