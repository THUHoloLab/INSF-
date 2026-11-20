#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
from insf import paths
from insf.simulation import generalization as W
from insf.baselines.mst import MSTNet, make_input
from insf.baselines.common import frame_scales
from insf.recon.single_frame import sync_time

ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--ckpt", type=Path, required=True)
args = ap.parse_args()
dev = W.dev
net = MSTNet().to(dev); ck = torch.load(args.ckpt, map_location=dev); net.load_state_dict(ck["model"]); net.eval()
for d in sorted((W.G / "out/runs").iterdir()):
    if not (d / "DONE").exists() or (d / f"{args.tag}_bands.npy").exists():
        continue
    meta = json.loads((d / "meta.json").read_text())
    op = W.operator(meta["cond"])
    y_np = np.load(d / "y.npy"); y = torch.from_numpy(y_np.reshape(-1)).to(dev)
    ms, _ = frame_scales(y_np, op["mask_sum"])
    yn = (y / ms).reshape(1, 512, 512); ones = torch.ones_like(yn)
    t0 = sync_time()
    with torch.no_grad():
        inp, x0 = make_input(yn, ones, op["a"][None]); x = net(inp, op["a"][None], x0)[0]
    secs = sync_time() - t0
    cube = (x.reshape(W.NK, -1).T * (ms / op["c"])) @ W.Bm
    bidx = torch.as_tensor(np.arange(meta["nm"][0], meta["nm"][1] + 1, 10) - 350, device=dev)
    np.save(d / f"{args.tag}_bands.npy", cube[:, bidx].reshape(512, 512, -1).cpu().numpy().astype(np.float16))
    np.save(d / f"{args.tag}_rgb.npy", (cube @ W.cie).reshape(512, 512, 3).cpu().numpy().astype(np.float32))
    meta["methods"][args.tag] = {"seconds": secs, "rmse_measurement": float(((cube * op["M601"]).sum(-1) - y).square().mean().sqrt()),
                                 "checkpoint": str(args.ckpt), "steps": ck["step"]}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    print(d.name, args.tag, round(meta["methods"][args.tag]["rmse_measurement"], 3), flush=True)
