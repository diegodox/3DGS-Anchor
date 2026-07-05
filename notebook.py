# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.13",
#     "gsplat==1.5.3",
#     "matplotlib==3.11.0",
#     "numpy==2.5.1",
#     "plyfile==1.1.4",
#     "torch==2.12.1",
# ]
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import matplotlib.pyplot as plt
    import gs_anchor as gsa


    return gsa, mo, plt, torch


@app.cell
def claude_status(mo):
    mo.md("""
    **Claude connected** 🤝 — ready to pair on this notebook.
    """)
    return


@app.cell
def hyperparams(mo, torch):
    F_DIM = 32
    K = 10
    VOXEL_SIZE = 2.0
    LR = 1e-2
    N_ITERS = 15000
    OPACITY_THRESHOLD = 0.005
    RENDER_SIZE = 96
    N_VIEWS = 16
    SUBSAMPLE_N = 30000
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mo.md(
        f"""
        **Hyperparameters**


        - Anchor feature dim `F_DIM` = {F_DIM}
        - Neighbors per anchor `K` = {K}
        - Voxel size = {VOXEL_SIZE}
        - Learning rate = {LR}
        - Training iterations = {N_ITERS}
        - Final opacity threshold = {OPACITY_THRESHOLD}
        - Render resolution = {RENDER_SIZE}x{RENDER_SIZE}, {N_VIEWS} synthetic views
        - Real-data subsample count = {SUBSAMPLE_N}
        - Device = `{DEVICE}`
        """
    )

    return (
        DEVICE,
        F_DIM,
        K,
        LR,
        N_ITERS,
        N_VIEWS,
        OPACITY_THRESHOLD,
        RENDER_SIZE,
        SUBSAMPLE_N,
        VOXEL_SIZE,
    )


@app.cell
def synthetic_data(gsa, mo):
    gaussians = gsa.make_synthetic_gaussians(n=800, seed=0)
    mo.md(f"Synthetic scene: **{len(gaussians)}** Gaussians.")
    return (gaussians,)


@app.cell
def ply_loader(mo):
    ply_path = mo.ui.text(label="Path to a real trained-3DGS .ply (optional)", value="/marimo/data/truck_point_cloud.ply")
    ply_path

    return (ply_path,)


@app.cell
def active_gaussians_selector(
    DEVICE,
    SUBSAMPLE_N,
    gaussians,
    gsa,
    mo,
    ply_path,
):
    _path = ply_path.value.strip() or "/marimo/data/truck_point_cloud.ply"
    if _path:
        _loaded = gsa.load_ply_gaussians(_path)
        active_gaussians = gsa.subsample_gaussians(_loaded, SUBSAMPLE_N, mode="topk_opacity").to(DEVICE)
        _source = f"loaded from `{_path}` ({len(_loaded)} -> {len(active_gaussians)} after subsample)"
    else:
        active_gaussians = gaussians.to(DEVICE)
        _source = "synthetic"

    mo.md(f"Active scene: **{len(active_gaussians)}** Gaussians ({_source}).")

    return (active_gaussians,)


@app.cell
def anchors(F_DIM, K, VOXEL_SIZE, active_gaussians, gsa, mo):
    anchors, anchor_coverage_mask = gsa.build_anchors(active_gaussians, VOXEL_SIZE, K, F_DIM)
    _coverage = anchor_coverage_mask.float().mean().item()
    mo.md(
        f"Built **{len(anchors)}** anchors from {len(active_gaussians)} Gaussians "
        f"(voxel size {VOXEL_SIZE}). Neighbor-slot coverage at init: {_coverage:.1%}."
    )
    return (anchors,)


@app.cell
def cameras_and_targets(
    N_VIEWS,
    RENDER_SIZE,
    active_gaussians,
    gsa,
    mo,
    torch,
):
    _scene_centroid = active_gaussians.positions.mean(dim=0)
    _scene_radius = 3.0 * active_gaussians.positions.std(dim=0).norm().clamp_min(0.5)
    cameras, camera_view_dirs = gsa.make_synthetic_cameras(
        _scene_centroid, _scene_radius, N_VIEWS, RENDER_SIZE
    )
    CANONICAL_DISTANCE = float(_scene_radius)

    with torch.no_grad():
        target_renders = [gsa.render(active_gaussians, cam) for cam in cameras]

    mo.md(f"Generated **{len(cameras)}** synthetic cameras and precomputed target renders.")
    return CANONICAL_DISTANCE, camera_view_dirs, cameras, target_renders


@app.cell
def models(DEVICE, F_DIM, K, gsa, mo):
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
    LR,
    N_ITERS,
    anchors,
    camera_view_dirs,
    cameras,
    color_mlp,
    cov_mlp,
    gsa,
    mo,
    opacity_mlp,
    target_renders,
):
    loss_history = gsa.train(
        anchors, opacity_mlp, color_mlp, cov_mlp,
        cameras, camera_view_dirs, target_renders,
        CANONICAL_DISTANCE, N_ITERS, LR,
    )
    mo.md(f"Training done. Final loss: **{loss_history[-1]:.5f}** (started at {loss_history[0]:.5f}).")
    return (loss_history,)


@app.cell
def final_decode(
    CANONICAL_DISTANCE,
    DEVICE,
    K,
    OPACITY_THRESHOLD,
    anchors,
    color_mlp,
    cov_mlp,
    gsa,
    mo,
    opacity_mlp,
    torch,
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
    gsa,
    loss_history,
    mo,
    reconstructed,
    target_renders,
):
    mean_l1, mean_mse = gsa.photometric_error_stats(reconstructed, cameras, target_renders)

    metrics = {
        "original Gaussians (N)": len(active_gaussians),
        "anchors (M)": len(anchors),
        "max candidates (M*K)": len(anchors) * K,
        "reconstructed Gaussians (N')": len(reconstructed),
        "mean photometric L1 across views": mean_l1,
        "mean photometric MSE across views": mean_mse,
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
    gsa,
    loss_history,
    plt,
    reconstructed,
    target_renders,
    torch,
):
    def plot_comparison(original, reconstructed, anchors, cameras, target_renders, loss_history):
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
        ax4.plot(loss_history)
        ax4.set_title("Training loss (photometric L1)")
        ax4.set_xlabel("iteration")

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


    plot_comparison(active_gaussians, reconstructed, anchors, cameras, target_renders, loss_history)

    return


if __name__ == "__main__":
    app.run()
