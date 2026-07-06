# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.13",
#     "gsplat==1.5.3",
#     "matplotlib==3.11.0",
#     "numpy==2.5.1",
#     "nvidia-cuda-cccl==13.0.*",
#     "nvidia-cuda-crt==13.0.*",
#     "nvidia-cuda-nvcc==13.0.*",
#     "nvidia-cuda-runtime==13.0.*",
#     "nvidia-nvvm==13.0.*",
#     "plyfile==1.1.4",
#     "torch==2.12.1",
# ]
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import torch
    import matplotlib.pyplot as plt
    import gs_anchor as gsa


@app.cell
def hyperparams():
    F_DIM = 32
    K = 10
    N_ITERS = 20000
    OPACITY_THRESHOLD = 0.005
    RENDER_SIZE = 96
    N_VIEWS = 24
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mo.md(
        f"""
        **Hyperparameters**


        - Anchor feature dim `F_DIM` = {F_DIM}
        - Neighbors per anchor `K` = {K}
        - Training iterations = {N_ITERS}
        - Final opacity threshold = {OPACITY_THRESHOLD}
        - Render resolution = {RENDER_SIZE}x{RENDER_SIZE}
        - Synthetic-fallback view count `N_VIEWS` = {N_VIEWS}
        - Device = `{DEVICE}`
        """
    )
    return DEVICE, F_DIM, K, N_ITERS, N_VIEWS, OPACITY_THRESHOLD, RENDER_SIZE


@app.cell
def synthetic_data():
    gaussians = gsa.make_synthetic_gaussians(n=800, seed=0)
    mo.md(f"Synthetic scene: **{len(gaussians)}** Gaussians.")
    return (gaussians,)


@app.cell
def ply_loader():
    ply_path = mo.ui.text(label="Path to a real trained-3DGS .ply (optional)", value="/marimo/data/truck_point_cloud.ply")
    ply_path
    return (ply_path,)


@app.cell
def active_gaussians_selector(DEVICE, gaussians, ply_path):
    _path = ply_path.value.strip() or "/marimo/data/truck_point_cloud.ply"
    if _path:
        active_gaussians = gsa.load_ply_gaussians(_path).to(DEVICE)
        _source = f"loaded from `{_path}` ({len(active_gaussians)} Gaussians, no crop, no subsample)"
    else:
        active_gaussians = gaussians.to(DEVICE)
        _source = "synthetic"

    mo.md(f"Active scene: **{len(active_gaussians)}** Gaussians ({_source}).")
    return (active_gaussians,)


@app.cell
def anchors(F_DIM, K, active_gaussians):
    VOXEL_SIZE = gsa.choose_voxel_size(active_gaussians.positions, K, target_ratio=1.0)
    anchors, anchor_coverage_mask = gsa.build_anchors(active_gaussians, VOXEL_SIZE, K, F_DIM)
    _coverage = anchor_coverage_mask.float().mean().item()
    mo.md(
        f"Built **{len(anchors)}** anchors from {len(active_gaussians)} Gaussians "
        f"(voxel size {VOXEL_SIZE:.4f}, chosen so M*K ≈ N). Neighbor-slot coverage at init: {_coverage:.1%}."
    )
    return (anchors,)


@app.cell
def cameras_and_targets(
    N_VIEWS,
    RENDER_SIZE,
    active_gaussians,
    real_cameras,
    real_view_dirs,
):
    if real_cameras is not None:
        cameras = real_cameras
        camera_view_dirs = real_view_dirs.to(active_gaussians.positions.device)
        _centroid_cpu = active_gaussians.positions.mean(dim=0).cpu()
        CANONICAL_DISTANCE = float(torch.stack([
            (-cam.R.T @ cam.t - _centroid_cpu).norm() for cam in cameras
        ]).mean())
        _source = f"{len(cameras)} real camera poses (COLMAP)"
    else:
        cameras, camera_view_dirs, _dists = gsa.make_random_cameras(
            active_gaussians.positions, N_VIEWS, RENDER_SIZE, seed=0
        )
        CANONICAL_DISTANCE = float(_dists.mean())
        _source = f"{len(cameras)} randomly generated cameras"

    with torch.no_grad():
        target_renders = [gsa.render(active_gaussians, cam) for cam in cameras]

    mo.md(f"Using **{_source}** for training (canonical distance {CANONICAL_DISTANCE:.2f}).")
    return CANONICAL_DISTANCE, camera_view_dirs, cameras, target_renders


@app.cell
def models(DEVICE, F_DIM, K):
    opacity_mlp = gsa.OpacityMLP(F_DIM, K).to(DEVICE)
    color_mlp = gsa.ColorMLP(F_DIM, K).to(DEVICE)
    cov_mlp = gsa.CovMLP(F_DIM, K).to(DEVICE)

    mo.md(
        f"Instantiated `opacity_mlp`, `color_mlp`, `cov_mlp` "
        f"(feature dim {F_DIM}, {K} neighbors per anchor) on `{DEVICE}`."
    )
    return color_mlp, cov_mlp, opacity_mlp


@app.cell
def training_loop(
    CANONICAL_DISTANCE,
    N_ITERS,
    anchors,
    camera_view_dirs,
    cameras,
    color_mlp,
    cov_mlp,
    opacity_mlp,
    target_renders,
):
    loss_history, psnr_history = gsa.train(
        anchors, opacity_mlp, color_mlp, cov_mlp,
        cameras, camera_view_dirs, target_renders,
        CANONICAL_DISTANCE, N_ITERS,
    )
    mo.md(
        f"Training done. Final loss: **{loss_history[-1]:.5f}**, "
        f"final PSNR: **{psnr_history[-1]:.2f} dB** (started at loss {loss_history[0]:.5f})."
    )
    return loss_history, psnr_history


@app.cell
def final_decode(
    CANONICAL_DISTANCE,
    DEVICE,
    K,
    OPACITY_THRESHOLD,
    anchors,
    color_mlp,
    cov_mlp,
    opacity_mlp,
):
    CANONICAL_VIEW_DIR = torch.tensor([0.0, 0.0, 1.0], device=DEVICE)

    with torch.no_grad():
        reconstructed = gsa.decode_to_gaussians(
            anchors, opacity_mlp, color_mlp, cov_mlp,
            CANONICAL_VIEW_DIR, CANONICAL_DISTANCE,
            hard_filter=True, threshold=OPACITY_THRESHOLD,
        )

    mo.md(
        f"Final structured -> standard decode (fixed view direction "
        f"`{CANONICAL_VIEW_DIR.tolist()}`): **{len(reconstructed)}** Gaussians "
        f"survive the opacity threshold (from {len(anchors) * K} candidates)."
    )
    return (reconstructed,)


@app.cell
def verification_metrics(
    K,
    active_gaussians,
    anchors,
    cameras,
    loss_history,
    psnr_history,
    reconstructed,
    target_renders,
):
    mean_l1, mean_mse, mean_psnr = gsa.photometric_error_stats(reconstructed, cameras, target_renders)

    metrics = {
        "original Gaussians (N)": len(active_gaussians),
        "anchors (M)": len(anchors),
        "max candidates (M*K)": len(anchors) * K,
        "reconstructed Gaussians (N')": len(reconstructed),
        "mean photometric L1 across views": mean_l1,
        "mean photometric MSE across views": mean_mse,
        "mean PSNR across views (dB)": mean_psnr,
        "final training PSNR (dB)": psnr_history[-1],
        "final training loss": loss_history[-1],
    }

    mo.vstack([
        mo.md("### Round-trip verification"),
        mo.ui.table([{"metric": k, "value": f"{v:.4f}" if isinstance(v, float) else v} for k, v in metrics.items()]),
    ])
    return


@app.cell
def visualization(
    active_gaussians,
    anchors,
    cameras,
    loss_history,
    psnr_history,
    reconstructed,
    target_renders,
):
    def plot_comparison(original, reconstructed, anchors, cameras, target_renders, loss_history, psnr_history):
        fig = plt.figure(figsize=(11, 6))

        ax1 = fig.add_subplot(2, 3, 1, projection="3d")
        orig_pos = original.positions.detach().cpu().numpy()
        orig_col = original.sh_dc.clamp(0, 1).detach().cpu().numpy()
        ax1.scatter(orig_pos[:, 0], orig_pos[:, 1], orig_pos[:, 2], c=orig_col, s=4)
        ax1.set_title("Original (standard 3DGS)")

        ax2 = fig.add_subplot(2, 3, 2, projection="3d")
        recon_pos = reconstructed.positions.detach().cpu().numpy()
        recon_col = reconstructed.sh_dc.clamp(0, 1).detach().cpu().numpy()
        ax2.scatter(recon_pos[:, 0], recon_pos[:, 1], recon_pos[:, 2], c=recon_col, s=4)
        ax2.set_title("Reconstructed (from anchors)")

        ax3 = fig.add_subplot(2, 3, 3, projection="3d")
        anchor_pos = anchors.anchor_positions.detach().cpu().numpy()
        ax3.scatter(anchor_pos[:, 0], anchor_pos[:, 1], anchor_pos[:, 2], c="gray", s=8)
        ax3.set_title(f"Anchors (M={len(anchors)})")

        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(loss_history, color="tab:blue")
        ax4.set_title("Training loss (L1) / PSNR")
        ax4.set_xlabel("iteration")
        ax4.set_ylabel("L1 loss", color="tab:blue")
        ax4b = ax4.twinx()
        ax4b.plot(psnr_history, color="tab:orange", alpha=0.6)
        ax4b.set_ylabel("PSNR (dB)", color="tab:orange")

        ax5 = fig.add_subplot(2, 3, 5)
        ax5.imshow(target_renders[0].detach().cpu().numpy())
        ax5.set_title("Target render (view 0)")
        ax5.axis("off")

        ax6 = fig.add_subplot(2, 3, 6)
        with torch.no_grad():
            recon_view0 = gsa.render(reconstructed, cameras[0])
        ax6.imshow(recon_view0.detach().cpu().numpy())
        ax6.set_title("Reconstructed render (view 0)")
        ax6.axis("off")

        fig.tight_layout()
        return fig


    plot_comparison(active_gaussians, reconstructed, anchors, cameras, target_renders, loss_history, psnr_history)
    return


@app.cell
def colmap_path():
    colmap_path = mo.ui.text(label="Path to a COLMAP sparse/0 dir with known camera poses (optional)", value="/marimo/data/truck_sparse")
    colmap_path
    return (colmap_path,)


@app.cell
def real_cameras_loader(RENDER_SIZE, colmap_path):
    import os as _os
    _colmap_dir = colmap_path.value.strip()
    if _colmap_dir and _os.path.exists(_os.path.join(_colmap_dir, "cameras.bin")):
        real_cameras, real_view_dirs = gsa.load_colmap_cameras(_colmap_dir, RENDER_SIZE)
        _status = mo.md(f"Loaded **{len(real_cameras)}** real camera poses from `{_colmap_dir}`.")
    else:
        real_cameras, real_view_dirs = None, None
        _status = mo.md("No COLMAP pose data found -- candidate cameras will be randomly generated.")
    _status
    return real_cameras, real_view_dirs


if __name__ == "__main__":
    app.run()
