import math
import time

import numpy as np
import torch

SATURATION = 1023.0


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def charbonnier(values: torch.Tensor, eps: float) -> torch.Tensor:
    return (torch.sqrt(values.square() + eps**2) - eps).mean()


def cosine_lr(step: int, total: int, lr0: float, lr_min: float) -> float:
    progress = step / max(total - 1, 1)
    return lr_min + 0.5 * (1.0 + math.cos(math.pi * progress)) * (lr0 - lr_min)


def make_xy(h: int, w: int, device) -> torch.Tensor:
    ys = (torch.arange(h, device=device) + 0.5) / h
    xs = (torch.arange(w, device=device) + 0.5) / w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2).float().contiguous()


def frame_scales(y: np.ndarray, mask_sum: np.ndarray) -> tuple[float, float]:
    valid = mask_sum > 1e-6
    meas = max(float(np.percentile(np.abs(y), 99)), 1e-6)
    sig = max(float(np.median(np.abs(y[valid] / mask_sum[valid]))), 1e-6)
    return meas, sig


@torch.inference_mode()
def render_cube(model, coords, wl, pixel_batch=8192) -> torch.Tensor:
    model.eval()
    out = torch.empty(coords.shape[0], wl.numel(), device=coords.device)
    for a in range(0, coords.shape[0], pixel_batch):
        out[a:a + pixel_batch] = model(coords[a:a + pixel_batch], wl)
    model.train()
    return out
