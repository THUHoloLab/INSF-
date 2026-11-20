#!/usr/bin/env python3
import argparse, json, sys, time, math
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from insf import paths
from insf.recon.single_frame import (Config, build_2d, set_seed, make_xy, charbonnier, cosine_lr, frame_scales,
                       render_cube, cie_matrix, WAVELENGTHS_NM)
HERE = Path(__file__).resolve().parent
R = paths.OUT / "colorchecker"
from insf.data.qe import channel_map
ap = argparse.ArgumentParser()
ap.add_argument("--key", required=True); ap.add_argument("--delta", type=int, default=0)
ap.add_argument("--black", type=float, default=0.0); ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--holdout", type=float, default=0.1); ap.add_argument("--size", type=int, default=640)
ap.add_argument("--tag", default="e1"); ap.add_argument("--save_cube", type=int, default=1)
a = ap.parse_args()
cfg = Config(); dev = torch.device("cuda")
f = json.load(open(paths.COLORCHECKER_FRAMES))[a.key]; r0, c0 = f["roi"]; H = W = a.size
raw = np.fromfile(R / "raw" / f"{a.key}_d{a.delta:+d}.raw", dtype="<u2").reshape(1200, 1600)[r0:r0+H, c0:c0+W].astype(np.float32)
qe = np.load(paths.DATA / "mask" / "qe.npy"); Q = qe[:, 1:].astype(np.float32)
chan = channel_map(H, W, r0, c0)
mask = Q.T[chan]
sat = raw >= 1023
y = np.clip(raw - a.black, 0, None) * cfg.measurement_gain
out = R / "runs" / f"{a.tag}_{a.key}_d{a.delta:+d}"; out.mkdir(parents=True, exist_ok=True)
log_f = open(out / "train_log.txt", "w")
def log(s):
    print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
rng = np.random.default_rng(20261116); hold = rng.random(H * W) < a.holdout
train_idx = torch.from_numpy(np.nonzero(~hold)[0]).to(dev)
mask_t = torch.from_numpy(mask.reshape(-1, 601)).to(dev); y_t = torch.from_numpy(y.reshape(-1)).to(dev)
meas_scale, sig_scale = frame_scales(y, mask.sum(-1))
xy = make_xy(H, W, dev); wl = torch.linspace(0, 1, 601, device=dev)
set_seed(cfg.seed); model = build_2d(cfg, sig_scale, dev); opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
n = train_idx.numel(); steps = n // cfg.batch_size; total = a.epochs * steps; k = 0; t0 = time.time()
for ep in range(a.epochs):
    perm = train_idx[torch.randperm(n, device=dev)]; acc = torch.zeros(3, device=dev)
    for s in range(steps):
        lr = cosine_lr(k, total, cfg.learning_rate, cfg.min_learning_rate)
        for g in opt.param_groups: g["lr"] = lr
        idx = perm[s*cfg.batch_size:(s+1)*cfg.batch_size]
        opt.zero_grad(set_to_none=True)
        spec = model(xy[idx], wl); est = (spec * mask_t[idx]).sum(-1)
        data = F.smooth_l1_loss(est / meas_scale, y_t[idx] / meas_scale, beta=cfg.huber_beta)
        stv = charbonnier(torch.diff(spec, dim=-1) / sig_scale, cfg.tv_epsilon)
        band = wl[torch.randint(wl.numel(), (1,), device=dev)]
        img = model(xy, band).reshape(H, W)
        ptv = 0.5 * (charbonnier((img[:, 1:] - img[:, :-1]) / sig_scale, cfg.tv_epsilon)
                     + charbonnier((img[1:] - img[:-1]) / sig_scale, cfg.tv_epsilon))
        loss = data + cfg.spectral_tv_weight * stv + cfg.spatial_tv_weight * ptv
        loss.backward(); opt.step(); acc += torch.stack([data.detach(), stv.detach(), ptv.detach()]); k += 1
    if ep % 10 == 9 or ep == 0:
        v = (acc / steps).tolist(); log(f"epoch {ep+1}/{a.epochs} data={v[0]:.5g} stv={v[1]:.5g} ptv={v[2]:.5g} t={time.time()-t0:.0f}s")
fit_s = time.time() - t0
cube = render_cube(model, xy, wl)
pred = (cube * mask_t).sum(-1)
res = (pred - y_t).cpu().numpy(); ok = ~sat.reshape(-1)
m = dict(key=a.key, delta=a.delta, black=a.black, epochs=a.epochs, fit_seconds=fit_s, roi=[r0, c0, H, W],
         rmse_train=float(np.sqrt((res[~hold]**2).mean())), rmse_holdout=float(np.sqrt((res[hold]**2).mean())),
         rmse_holdout_unsat=float(np.sqrt((res[hold & ok]**2).mean())), y_std=float(y.std()), sat_frac=float(sat.mean()),
         meas_scale=meas_scale, sig_scale=sig_scale)
cube_np = cube.reshape(H, W, 601).cpu().numpy()

def centers(c):
    TL, TR, BL, BR = [np.array(p, float) for p in c]; o = []
    for i in range(6):
        for j in range(4):
            u = j/3; v = i/5; o.append((1-v)*((1-u)*TL+u*TR)+v*((1-u)*BL+u*BR))
    return np.array(o)
hw = 12; P = []; Pstd = []; Y9 = []
for (x, yy) in centers(f["corners"]):
    px = int(round(3*x+1-c0)); py = int(round(3*yy+1-r0))
    blk = cube_np[py-hw:py+hw+1, px-hw:px+hw+1]; P.append(blk.mean((0, 1))); Pstd.append(blk.std((0, 1)))
    rb = (raw - a.black)[py-hw:py+hw+1, px-hw:px+hw+1]; cb = chan[py-hw:py+hw+1, px-hw:px+hw+1]
    Y9.append([rb[cb == c].mean() for c in range(9)])
tile = np.load(R / "tile" / f"{a.key}.npy")
T = cube_np[tile].mean(0); T9 = [(raw - a.black)[tile & (chan == c)].mean() for c in range(9)]
np.savez(out / "spectra.npz", patch=np.array(P), patch_std=np.array(Pstd), tile=T, patch9=np.array(Y9), tile9=np.array(T9), wl=WAVELENGTHS_NM)
rgbm = cie_matrix(paths.XYZBAR)
np.save(out / "rgb_linear.npy", (cube_np @ rgbm).astype(np.float32))
bi = [int(nm-350) for nm in (450, 500, 550, 600, 650, 700, 750, 800, 850, 900)]
np.save(out / "bands.npy", cube_np[:, :, bi].astype(np.float32))
np.save(out / "pred_measurement.npy", pred.reshape(H, W).cpu().numpy().astype(np.float32)); np.save(out / "measurement.npy", y)
if a.save_cube: np.save(out / "cube_f16.npy", cube_np.astype(np.float16))
json.dump(m, open(out / "metrics.json", "w"), indent=1); log(json.dumps(m))
