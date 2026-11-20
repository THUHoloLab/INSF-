#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
from insf import paths
from insf.simulation.datasets import G, to601
from insf.baselines.common import NK, interp_matrix, frame_scales
from insf.baselines import twist, admm
from insf.recon.single_frame import (Config, build_2d, fit_frame, render_cube, make_xy, set_seed, sync_time, cie_matrix)
from insf.data.io import _load_mask

P99, GAIN = 336.0, 0.5
NOISE = {"base": (4.0, 2.0), "noise2x": (1.0, 4.0), "noise0.5x": (16.0, 1.0), "maskshift": (4.0, 2.0)}
TW = json.loads((paths.OUT / "baselines/runs/twist/tuning.json").read_text())["chosen"]
AD = json.loads((paths.OUT / "baselines/runs/admm/tuning.json").read_text())["chosen"]
MASK = paths.MASK
dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
cfg = Config(); cfg.crop_size = 512
cie = torch.from_numpy(cie_matrix(paths.XYZBAR)).to(dev)
Bm = torch.from_numpy(interp_matrix()).to(dev)
_ops = {}


def operator(cond):
    key = "shift" if cond == "maskshift" else "central"
    if key not in _ops:
        r0 = 199 if key == "shift" else 443
        sl = slice(r0, r0 + 512)
        m = _load_mask(MASK, "dataCube", sl, sl)
        M601 = torch.from_numpy(m.reshape(-1, 601)).to(dev)
        M61 = M601 @ Bm.T
        c = float(M61.norm(dim=1).max())
        _ops[key] = dict(M601=M601, c=c, a=(M61 / c).T.reshape(NK, 512, 512).contiguous(),
                         mask_sum=m.sum(-1), rows_1based=[r0 + 1, r0 + 512])
    return _ops[key]


def main():
    q = json.loads((G / "queue.json").read_text())["queue"]
    for qi, (ds, name, cond) in enumerate(q):
        d = G / "out/runs" / f"{ds}_{name}_{cond}"
        try:
            os.makedirs(d / ".claim")
        except FileExistsError:
            continue
        op = operator(cond)
        z = np.load(G / "out/gt" / f"{ds}_{name}.npz")
        gt31, nm = z["gt31"].astype(np.float32), z["nm"]
        gt = torch.from_numpy(to601(gt31, nm)).to(dev).reshape(-1, 601)
        y0 = (gt * op["M601"]).sum(-1)
        s = P99 / float(torch.quantile(y0[::7], 0.99))
        gt, y0 = gt * s, y0 * s
        K, sr = NOISE[cond]
        rng = np.random.default_rng(20261020 + qi)
        raw = y0.cpu().numpy() / GAIN
        raw = rng.poisson(np.clip(raw, 0, None) * K) / K + rng.normal(0, sr, raw.shape)
        raw = np.clip(np.round(raw), 0, 1023).astype(np.float32)
        y_np = (raw * GAIN).reshape(512, 512)
        np.save(d / "y.npy", y_np)
        bidx = torch.as_tensor(nm - 350, device=dev)
        np.save(d / "gt_rgb.npy", (gt @ cie).reshape(512, 512, 3).cpu().numpy().astype(np.float32))
        meta = {"ds": ds, "name": name, "cond": cond, "queue_index": qi, "scale": s, "K": K, "read_sigma": sr,
                "mask_rows_cols_1based": op["rows_1based"], "noise_rmse": float(np.sqrt(((y_np.reshape(-1) - y0.cpu().numpy()) ** 2).mean())),
                "saturated_fraction": float((raw >= 1023).mean()), "nm": [int(nm[0]), int(nm[-1])], "methods": {}}
        y = torch.from_numpy(y_np.reshape(-1)).to(dev)
        ms, sig = frame_scales(y_np, op["mask_sum"])
        yn = (y / ms).reshape(512, 512)
        ones = torch.ones(512, 512, device=dev)
        a, c, M601 = op["a"], op["c"], op["M601"]

        def save(m, cube, secs):
            res = (cube * M601).sum(-1) - y
            np.save(d / f"{m}_bands.npy", cube[:, bidx].reshape(512, 512, -1).cpu().numpy().astype(np.float16))
            np.save(d / f"{m}_rgb.npy", (cube @ cie).reshape(512, 512, 3).cpu().numpy().astype(np.float32))
            meta["methods"][m] = {"seconds": secs, "rmse_measurement": float(res.square().mean().sqrt())}
            (d / "meta.json").write_text(json.dumps(meta, indent=2))
        k2c = lambda x: (x.reshape(NK, -1).T * (ms / c)) @ Bm
        t0 = sync_time(); x = twist.solve(a, yn, ones, TW["lam"], TW["mu"], iters=400, inner=10); save("twist", k2c(x), sync_time() - t0)
        t0 = sync_time(); x = admm.solve(a, yn, ones, AD["lam"], AD["mu"], iters=150, cg_iters=8); save("admm", k2c(x), sync_time() - t0)
        xy = make_xy(512, 512, dev); wl = torch.linspace(0.0, 1.0, 601, device=dev)
        set_seed(cfg.seed); model = build_2d(cfg, sig, dev); opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        t0 = sync_time()
        fit_frame(model, opt, xy, wl, M601, y, ms, sig, 512, 512, cfg, cfg.epochs, cfg.learning_rate, cfg.min_learning_rate, lambda s_: None)
        save("inr", render_cube(model, xy, wl), sync_time() - t0)
        np.save(d / "gt_bands.npy", gt[:, bidx].reshape(512, 512, -1).cpu().numpy().astype(np.float16))
        (d / "DONE").write_text("ok")
        print(f"[{time.strftime('%H:%M:%S')}] {ds} {name} {cond} " + " ".join(f"{m}:{v['rmse_measurement']:.2f}/{v['seconds']:.0f}s" for m, v in meta["methods"].items()), flush=True)
        del model, opt, gt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
