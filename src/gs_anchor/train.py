import math

import torch
import torch.nn.functional as F

from .model import ColorMLP, CovMLP, OpacityMLP, decode_to_gaussians
from .render import render


def _psnr(mse: float) -> float:
    """PSNR in dB against a [0,1]-normalized image (MAX=1)."""
    return 10.0 * math.log10(1.0 / mse) if mse > 0 else float("inf")


def train(
    anchors,
    opacity_mlp: OpacityMLP,
    color_mlp: ColorMLP,
    cov_mlp: CovMLP,
    cameras,
    camera_view_dirs,
    target_renders,
    distance: float,
    n_iters: int,
    lr: float,
):
    params = [
        anchors.anchor_features,
        anchors.anchor_scaling,
        anchors.anchor_offsets,
        *opacity_mlp.parameters(),
        *color_mlp.parameters(),
        *cov_mlp.parameters(),
    ]
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_history = []
    psnr_history = []
    n_views = len(cameras)
    rng = torch.Generator().manual_seed(0)

    for it in range(n_iters):
        view_idx = torch.randint(0, n_views, (1,), generator=rng).item()
        cam = cameras[view_idx]
        view_dir = camera_view_dirs[view_idx]
        target_img = target_renders[view_idx]

        decoded = decode_to_gaussians(
            anchors, opacity_mlp, color_mlp, cov_mlp, view_dir, distance, hard_filter=False
        )
        pred_img = render(decoded, cam)
        loss = F.l1_loss(pred_img, target_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            psnr = _psnr(F.mse_loss(pred_img, target_img).item())

        loss_history.append(loss.item())
        psnr_history.append(psnr)
        if it % 100 == 0 or it == n_iters - 1:
            print(f"iter {it:5d}  loss {loss.item():.5f}  psnr {psnr:6.2f} dB")

    return loss_history, psnr_history


def photometric_error_stats(gaussians, cameras, targets):
    with torch.no_grad():
        renders = [render(gaussians, cam) for cam in cameras]
    l1 = [F.l1_loss(r, t).item() for r, t in zip(renders, targets)]
    l2 = [F.mse_loss(r, t).item() for r, t in zip(renders, targets)]
    psnr = [_psnr(v) for v in l2]
    return sum(l1) / len(l1), sum(l2) / len(l2), sum(psnr) / len(psnr)
