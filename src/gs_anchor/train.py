import torch
import torch.nn.functional as F

from .model import ColorMLP, CovMLP, OpacityMLP, decode_to_gaussians
from .render import render


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

        loss_history.append(loss.item())
        if it % 100 == 0 or it == n_iters - 1:
            print(f"iter {it:5d}  loss {loss.item():.5f}")

    return loss_history


def photometric_error_stats(gaussians, cameras, targets):
    with torch.no_grad():
        renders = [render(gaussians, cam) for cam in cameras]
    l1 = [F.l1_loss(r, t).item() for r, t in zip(renders, targets)]
    l2 = [F.mse_loss(r, t).item() for r, t in zip(renders, targets)]
    return sum(l1) / len(l1), sum(l2) / len(l2)
