import math

import torch
import torch.nn.functional as F

from .model import ColorMLP, CovMLP, OpacityMLP, decode_to_gaussians
from .render import render


def _psnr(mse: float) -> float:
    """PSNR in dB against a [0,1]-normalized image (MAX=1)."""
    return 10.0 * math.log10(1.0 / mse) if mse > 0 else float("inf")


def _decay_to(final_ratio: float, n_iters: int):
    """LambdaLR multiplier decaying geometrically from 1.0 to `final_ratio`."""
    def f(it):
        t = min(it / max(n_iters - 1, 1), 1.0)
        return final_ratio ** t
    return f


def _flat():
    return lambda it: 1.0


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
    lr_feature: float = 0.0075,
    lr_scaling: float = 0.007,
    lr_offset: float = 0.01,
    lr_opacity_mlp: float = 0.002,
    lr_color_mlp: float = 0.008,
    lr_cov_mlp: float = 0.004,
    scaling_reg_weight: float = 0.01,
):
    """Per-parameter-group learning rates and decay schedule, adapted from
    the Scaffold-GS reference implementation's ratios (that codebase uses
    a similar per-group split -- anchor offsets and most MLP weights decay
    over training, anchor features/scaling and the covariance MLP stay
    flat). A single flat lr for everything (including MLP weights, which
    the reference trains 2-3 orders of magnitude slower) was destabilizing
    the decoded per-Gaussian attributes."""
    optimizer = torch.optim.Adam([
        {"params": [anchors.anchor_features], "lr": lr_feature},
        {"params": [anchors.anchor_scaling], "lr": lr_scaling},
        {"params": [anchors.anchor_offsets], "lr": lr_offset},
        {"params": list(opacity_mlp.parameters()), "lr": lr_opacity_mlp},
        {"params": list(color_mlp.parameters()), "lr": lr_color_mlp},
        {"params": list(cov_mlp.parameters()), "lr": lr_cov_mlp},
    ])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[
        _flat(),                          # anchor_features
        _flat(),                          # anchor_scaling
        _decay_to(0.01, n_iters),         # anchor_offsets:  0.01    -> 1e-4
        _decay_to(0.01, n_iters),         # opacity_mlp:     0.002   -> 2e-5
        _decay_to(0.00625, n_iters),      # color_mlp:       0.008   -> 5e-5
        _flat(),                          # cov_mlp
    ])

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
        photo_loss = F.l1_loss(pred_img, target_img)
        scaling_reg = anchors.anchor_scaling.abs().mean()
        loss = photo_loss + scaling_reg_weight * scaling_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            psnr = _psnr(F.mse_loss(pred_img, target_img).item())

        loss_history.append(photo_loss.item())
        psnr_history.append(psnr)
        if it % 100 == 0 or it == n_iters - 1:
            print(f"iter {it:5d}  loss {photo_loss.item():.5f}  psnr {psnr:6.2f} dB")

    return loss_history, psnr_history


def photometric_error_stats(gaussians, cameras, targets):
    with torch.no_grad():
        renders = [render(gaussians, cam) for cam in cameras]
    l1 = [F.l1_loss(r, t).item() for r, t in zip(renders, targets)]
    l2 = [F.mse_loss(r, t).item() for r, t in zip(renders, targets)]
    psnr = [_psnr(v) for v in l2]
    return sum(l1) / len(l1), sum(l2) / len(l2), sum(psnr) / len(psnr)
