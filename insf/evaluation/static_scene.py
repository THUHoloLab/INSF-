#!/usr/bin/env python3
import csv, json, sys
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from insf import paths
from insf.recon.single_frame import cie_matrix

D = paths.OUT / "static_single_frame" / "sim_flowers"
P = D / "png"; P.mkdir(exist_ok=True)
dev = torch.device("cuda")
cie = torch.from_numpy(cie_matrix(paths.XYZBAR)).to(dev)
B31 = np.arange(400, 701, 10) - 350
gt = torch.from_numpy(np.load(D / "gt_cube.npy")).to(dev)
H, W, L = gt.shape
lin_gt = (gt.reshape(-1, L) @ cie).reshape(H, W, 3).clamp_min(0)
m = lin_gt.reshape(-1, 3).mean(0); gains = m.mean() / m
expo = float(torch.quantile((lin_gt * gains).reshape(-1)[::3], 0.995))


def disp(cube):
    lin = ((cube.reshape(-1, L) @ cie).reshape(H, W, 3).clamp_min(0) * gains / expo).clamp(0, 1)
    return torch.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin.clamp_min(1e-12) ** (1 / 2.4) - 0.055)


def ssim(a, b, rng=1.0):
    g = torch.exp(-((torch.arange(11, device=dev) - 5.0) ** 2) / (2 * 1.5 ** 2)); g = g / g.sum()
    w = (g[:, None] * g[None])[None, None].expand(a.shape[0], 1, 11, 11)
    f = lambda x: F.conv2d(x[None], w, groups=a.shape[0])[0]
    c1, c2 = (0.01 * rng) ** 2, (0.03 * rng) ** 2
    ma, mb = f(a), f(b)
    va, vb, cab = f(a * a) - ma ** 2, f(b * b) - mb ** 2, f(a * b) - ma * mb
    return float((((2 * ma * mb + c1) * (2 * cab + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))).mean())


def psnr(a, b, rng=1.0):
    return float(10 * torch.log10(rng ** 2 / (a - b).square().mean()))


g31 = gt[..., B31]; gmax = g31.max()
gn = (g31 / gmax).permute(2, 0, 1)
rgb_gt = disp(gt)
Image.fromarray((rgb_gt.cpu().numpy() * 255).round().astype(np.uint8)).save(P / "gt_full.png")
y = np.load(D / "measurement.npy")
Image.fromarray((np.clip(y / np.percentile(y, 99.5), 0, 1) * 255).round().astype(np.uint8)).convert("RGB").save(P / "measurement_full.png")
meta = json.loads((D / "meta.json").read_text())
rows = []
for k in ("twist", "admm", "mst", "inr"):
    cube = torch.from_numpy(np.load(D / f"{k}_cube.npy").astype(np.float32)).to(dev)
    c31 = cube[..., B31]
    rn = (c31 / gmax).permute(2, 0, 1)
    num = (c31 * g31).sum(-1); den = c31.norm(dim=-1) * g31.norm(dim=-1)
    ok = den > 1e-12
    sam = float(torch.rad2deg(torch.arccos((num[ok] / den[ok]).clamp(-1, 1))).mean())
    rgb = disp(cube)
    Image.fromarray((rgb.cpu().numpy() * 255).round().astype(np.uint8)).save(P / f"{k}_full.png")
    mm = meta["methods"][k]
    rows.append({"method": k, "psnr_rgb": psnr(rgb, rgb_gt), "ssim_rgb": ssim(rgb.permute(2, 0, 1), rgb_gt.permute(2, 0, 1)),
                 "psnr_cube_400_700": psnr(rn, gn), "ssim_cube_400_700": ssim(rn, gn), "sam_deg_400_700": sam,
                 "rmse_measurement": mm["rmse_measurement_all"], "seconds": mm["seconds"]})
    print(rows[-1], flush=True)
    del cube
with open(D / "metrics.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
(D / "metrics.json").write_text(json.dumps({"rows": rows, "white_balance_gains_from_gt": gains.tolist(), "exposure_from_gt": expo,
                                            "notes": __doc__}, indent=2))
