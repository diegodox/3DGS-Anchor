# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.13",
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
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from plyfile import PlyData, PlyElement
    from dataclasses import dataclass


    return F, PlyData, dataclass, mo, nn, np, plt, torch


@app.cell
def claude_status(mo):
    mo.md("""
    **Claude connected** 🤝 — ready to pair on this notebook.
    """)
    return


@app.cell
def hyperparams(mo):
    F_DIM = 32
    K = 10
    VOXEL_SIZE = 0.4
    LR = 1e-2
    N_ITERS = 800
    OPACITY_THRESHOLD = 0.005
    RENDER_SIZE = 48
    N_VIEWS = 10
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
        """
    )

    return (
        F_DIM,
        K,
        LR,
        N_ITERS,
        N_VIEWS,
        OPACITY_THRESHOLD,
        RENDER_SIZE,
        VOXEL_SIZE,
    )


@app.cell
def data_structures(dataclass, mo, torch):
    @dataclass
    class GaussianCloud:
        positions: torch.Tensor   # [N,3] float32
        scales: torch.Tensor      # [N,3] log-scale
        rotations: torch.Tensor   # [N,4] quaternion (unnormalized)
        opacities: torch.Tensor   # [N,1] logit
        sh_dc: torch.Tensor       # [N,3] degree-0 SH / base RGB
        sh_rest: torch.Tensor     # [N,45] higher-order SH (zero-filled if unused)

        def __len__(self):
            return self.positions.shape[0]


    @dataclass
    class AnchorCloud:
        anchor_positions: torch.Tensor  # [M,3] fixed
        anchor_features: torch.Tensor   # [M,F_DIM] nn.Parameter
        anchor_scaling: torch.Tensor    # [M,3] nn.Parameter
        anchor_offsets: torch.Tensor    # [M,K,3] nn.Parameter

        def __len__(self):
            return self.anchor_positions.shape[0]


    @dataclass
    class Camera:
        R: torch.Tensor       # [3,3] world->camera rotation
        t: torch.Tensor       # [3] world->camera translation
        focal: float
        H: int
        W: int

        @property
        def principal(self):
            return (self.W / 2.0, self.H / 2.0)


    mo.md("Defined `GaussianCloud`, `AnchorCloud`, `Camera`.")

    return AnchorCloud, Camera, GaussianCloud


@app.cell
def synthetic_data(GaussianCloud, mo, torch):
    def make_synthetic_gaussians(n=800, seed=0) -> GaussianCloud:
        g = torch.Generator().manual_seed(seed)
        # scatter points on a fuzzy sphere shell so there's real 3D structure
        dirs = torch.randn(n, 3, generator=g)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        radius = 1.0 + 0.15 * torch.randn(n, generator=g)
        positions = dirs * radius.unsqueeze(-1)

        scales = torch.log(0.03 + 0.02 * torch.rand(n, 3, generator=g))
        rotations = torch.randn(n, 4, generator=g)
        rotations = rotations / rotations.norm(dim=-1, keepdim=True)
        opacities = torch.logit(0.5 + 0.4 * torch.rand(n, 1, generator=g), eps=1e-4)

        # color as a function of direction, so there's real structure to learn
        sh_dc = 0.5 + 0.5 * dirs
        sh_rest = torch.zeros(n, 45)

        return GaussianCloud(
            positions=positions,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            sh_dc=sh_dc,
            sh_rest=sh_rest,
        )


    gaussians = make_synthetic_gaussians(n=800, seed=0)
    mo.md(f"Synthetic scene: **{len(gaussians)}** Gaussians.")

    return (gaussians,)


@app.cell
def ply_loader(GaussianCloud, PlyData, mo, np, torch):
    def load_ply_gaussians(path: str) -> GaussianCloud:
        """Load a standard trained-3DGS .ply into a GaussianCloud."""
        ply = PlyData.read(path)
        v = ply["vertex"]
        names = v.data.dtype.names

        positions = torch.tensor(
            np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
        )
        scales = torch.tensor(
            np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32)
        )
        rotations = torch.tensor(
            np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float32)
        )
        opacities = torch.tensor(v["opacity"].astype(np.float32)).unsqueeze(-1)
        sh_dc = torch.tensor(
            np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32)
        )
        rest_names = sorted(
            (n for n in names if n.startswith("f_rest_")),
            key=lambda n: int(n.split("_")[-1]),
        )
        if rest_names:
            sh_rest = torch.tensor(
                np.stack([v[n] for n in rest_names], axis=-1).astype(np.float32)
            )
        else:
            sh_rest = torch.zeros(positions.shape[0], 45)

        return GaussianCloud(
            positions=positions,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            sh_dc=sh_dc,
            sh_rest=sh_rest,
        )


    ply_path = mo.ui.text(label="Path to a real trained-3DGS .ply (optional)", value="")
    ply_path

    return load_ply_gaussians, ply_path


@app.cell
def active_gaussians_selector(gaussians, load_ply_gaussians, mo, ply_path):
    if ply_path.value.strip():
        active_gaussians = load_ply_gaussians(ply_path.value.strip())
        _source = f"loaded from `{ply_path.value.strip()}`"
    else:
        active_gaussians = gaussians
        _source = "synthetic"

    mo.md(f"Active scene: **{len(active_gaussians)}** Gaussians ({_source}).")

    return (active_gaussians,)


@app.cell
def build_anchors_cell(
    AnchorCloud,
    F_DIM,
    GaussianCloud,
    K,
    VOXEL_SIZE,
    active_gaussians,
    mo,
    nn,
    torch,
):
    def build_anchors(gaussians: GaussianCloud, voxel_size: float, k: int):
        """Voxelize Gaussian positions into anchors; warm-start scaling/offsets
        from the Gaussians assigned to each anchor. These assignments are used
        only for initialization, not as training targets (training is
        render-based, see the training-loop cell)."""
        positions = gaussians.positions
        voxel_idx = torch.floor(positions / voxel_size).to(torch.int64)
        unique_idx, inverse = torch.unique(voxel_idx, dim=0, return_inverse=True)
        anchor_positions = unique_idx.float() * voxel_size + voxel_size / 2
        M = anchor_positions.shape[0]

        order = torch.argsort(inverse)
        counts = torch.bincount(inverse, minlength=M)
        starts = torch.cumsum(counts, dim=0) - counts

        anchor_offsets_init = torch.zeros(M, k, 3)
        anchor_scaling_init = torch.full((M, 3), voxel_size / 2)
        coverage_mask = torch.zeros(M, k, dtype=torch.bool)

        _g = torch.Generator().manual_seed(0)
        for m in range(M):
            start = int(starts[m])
            count = int(counts[m])
            idxs = order[start:start + count]
            if count > k:
                perm = torch.randperm(count, generator=_g)[:k]
                idxs = idxs[perm]
            n_assigned = idxs.shape[0]
            if n_assigned == 0:
                continue
            assigned_pos = positions[idxs]
            spread = assigned_pos.std(dim=0, unbiased=False).clamp_min(voxel_size * 0.1)
            anchor_scaling_init[m] = spread
            anchor_offsets_init[m, :n_assigned] = (assigned_pos - anchor_positions[m]) / spread
            coverage_mask[m, :n_assigned] = True

        anchor_features = nn.Parameter(0.01 * torch.randn(M, F_DIM))
        anchor_scaling = nn.Parameter(anchor_scaling_init.clone())
        anchor_offsets = nn.Parameter(anchor_offsets_init + 0.01 * torch.randn(M, k, 3))

        anchors = AnchorCloud(
            anchor_positions=anchor_positions,
            anchor_features=anchor_features,
            anchor_scaling=anchor_scaling,
            anchor_offsets=anchor_offsets,
        )
        return anchors, coverage_mask


    anchors, anchor_coverage_mask = build_anchors(active_gaussians, VOXEL_SIZE, K)
    _coverage = anchor_coverage_mask.float().mean().item()
    mo.md(
        f"Built **{len(anchors)}** anchors from {len(active_gaussians)} Gaussians "
        f"(voxel size {VOXEL_SIZE}). Neighbor-slot coverage at init: {_coverage:.1%}."
    )

    return (anchors,)


@app.cell
def renderer(Camera, GaussianCloud, mo, torch):
    def render(gaussians: GaussianCloud, camera: Camera) -> torch.Tensor:
        """Differentiable, pure-PyTorch renderer: perspective-project each
        Gaussian, give it an isotropic image-space footprint (simplification --
        no anisotropic 2D-covariance/Jacobian projection), and alpha-composite
        front-to-back via cumulative transmittance. Returns an [H,W,3] image."""
        positions = gaussians.positions
        p_cam = positions @ camera.R.T + camera.t
        z = p_cam[:, 2].clamp_min(1e-3)
        x_img = camera.focal * p_cam[:, 0] / z + camera.principal[0]
        y_img = camera.focal * p_cam[:, 1] / z + camera.principal[1]

        mean_scale = torch.exp(gaussians.scales).mean(dim=-1)
        radius_px = (camera.focal * mean_scale / z).clamp_min(0.5)

        opacity = torch.sigmoid(gaussians.opacities.squeeze(-1))
        color = torch.sigmoid(gaussians.sh_dc)

        ys, xs = torch.meshgrid(
            torch.arange(camera.H, dtype=torch.float32),
            torch.arange(camera.W, dtype=torch.float32),
            indexing="ij",
        )
        dx = xs.unsqueeze(0) - x_img.view(-1, 1, 1)
        dy = ys.unsqueeze(0) - y_img.view(-1, 1, 1)
        dist2 = dx * dx + dy * dy
        alpha = opacity.view(-1, 1, 1) * torch.exp(-0.5 * dist2 / (radius_px.view(-1, 1, 1) ** 2))

        order = torch.argsort(z)
        alpha_sorted = alpha[order]
        color_sorted = color[order]

        transmittance = torch.cumprod(
            torch.cat([torch.ones(1, camera.H, camera.W), 1 - alpha_sorted + 1e-10], dim=0),
            dim=0,
        )[:-1]
        weight = transmittance * alpha_sorted
        image = torch.einsum("nhw,nc->hwc", weight, color_sorted)
        return image.clamp(0.0, 1.0)


    def look_at_camera(eye: torch.Tensor, target: torch.Tensor, focal: float, size: int) -> Camera:
        forward = (target - eye)
        forward = forward / forward.norm()
        up_hint = torch.tensor([0.0, 1.0, 0.0])
        if torch.abs(torch.dot(forward, up_hint)) > 0.99:
            up_hint = torch.tensor([1.0, 0.0, 0.0])
        right = torch.linalg.cross(forward, up_hint)
        right = right / right.norm()
        up = torch.linalg.cross(right, forward)
        # world->camera rotation: rows are camera axes (right, up, -forward... use +forward as +z)
        R = torch.stack([right, up, forward], dim=0)
        t = -R @ eye
        return Camera(R=R, t=t, focal=focal, H=size, W=size)


    mo.md("Defined `render()` and `look_at_camera()`.")

    return look_at_camera, render


@app.cell
def cameras_and_targets(
    N_VIEWS,
    RENDER_SIZE,
    active_gaussians,
    look_at_camera,
    mo,
    render,
    torch,
):
    def make_synthetic_cameras(centroid: torch.Tensor, radius: float, n_views: int, size: int):
        cams = []
        dirs_list = []
        golden_angle = torch.pi * (3.0 - 5.0 ** 0.5)
        for i in range(n_views):
            yy = 1 - 2 * (i / max(n_views - 1, 1))
            r = (1 - yy * yy) ** 0.5
            theta = golden_angle * i
            x = torch.cos(torch.tensor(theta)) * r
            z = torch.sin(torch.tensor(theta)) * r
            d = torch.tensor([x, yy, z])
            eye = centroid + radius * d
            cams.append(look_at_camera(eye, centroid, focal=1.2 * size, size=size))
            dirs_list.append(-d)
        return cams, torch.stack(dirs_list)


    _scene_centroid = active_gaussians.positions.mean(dim=0)
    _scene_radius = 3.0 * active_gaussians.positions.std(dim=0).norm().clamp_min(0.5)
    cameras, camera_view_dirs = make_synthetic_cameras(
        _scene_centroid, _scene_radius, N_VIEWS, RENDER_SIZE
    )
    CANONICAL_DISTANCE = float(_scene_radius)

    with torch.no_grad():
        target_renders = [render(active_gaussians, cam) for cam in cameras]

    mo.md(f"Generated **{len(cameras)}** synthetic cameras and precomputed target renders.")

    return CANONICAL_DISTANCE, camera_view_dirs, cameras, target_renders


@app.cell
def mlps(F_DIM, K, mo, nn, torch):
    MLP_INPUT_DIM = F_DIM + 3 + 1  # anchor feature + view direction + distance


    class OpacityMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(MLP_INPUT_DIM, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, K),
            )

        def forward(self, x):
            return self.net(x)  # raw logits, [.., K]


    class ColorMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(MLP_INPUT_DIM, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, K * 3),
            )

        def forward(self, x):
            return torch.sigmoid(self.net(x))  # RGB in [0,1], [.., K*3]


    class CovMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(MLP_INPUT_DIM, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, K * 7),
            )

        def forward(self, x):
            return self.net(x)  # [.., K*7]: 3 scale-offset + 4 quat-offset per neighbor


    opacity_mlp = OpacityMLP()
    color_mlp = ColorMLP()
    cov_mlp = CovMLP()

    mo.md(
        f"Defined `OpacityMLP`, `ColorMLP`, `CovMLP` (input dim {MLP_INPUT_DIM}) "
        f"and instantiated `opacity_mlp`, `color_mlp`, `cov_mlp`."
    )

    return color_mlp, cov_mlp, opacity_mlp


@app.cell
def decode_fn(
    AnchorCloud,
    GaussianCloud,
    K,
    OPACITY_THRESHOLD,
    color_mlp,
    cov_mlp,
    mo,
    opacity_mlp,
    torch,
):
    def decode_to_gaussians(
        anchors: AnchorCloud,
        view_dir: torch.Tensor,
        distance: float,
        hard_filter: bool = False,
        threshold: float = OPACITY_THRESHOLD,
    ) -> GaussianCloud:
        """Decode anchors+MLPs into a flat GaussianCloud for a given viewing
        direction/distance. `hard_filter=False` (training mode) keeps all M*K
        soft-opacity candidates so the photometric loss can shape them via
        compositing. `hard_filter=True` (final structured->standard decode)
        drops near-zero-opacity candidates, matching the paper's pruning."""
        M = len(anchors)
        scaling = anchors.anchor_scaling.clamp_min(1e-4)
        view_dir = view_dir / view_dir.norm()
        view_dir_b = view_dir.view(1, 3).expand(M, 3)
        distance_b = torch.full((M, 1), float(distance))
        mlp_in = torch.cat([anchors.anchor_features, view_dir_b, distance_b], dim=-1)

        opacity_logits = opacity_mlp(mlp_in)               # [M,K]
        color = color_mlp(mlp_in).view(M, K, 3)             # [M,K,3]
        cov_out = cov_mlp(mlp_in).view(M, K, 7)             # [M,K,7]
        scale_offset = cov_out[..., :3]
        quat_offset = cov_out[..., 3:7]

        positions = anchors.anchor_positions.unsqueeze(1) + anchors.anchor_offsets * scaling.unsqueeze(1)
        scales = torch.log(scaling).unsqueeze(1) + scale_offset
        rotations = quat_offset
        opacities = opacity_logits.unsqueeze(-1)

        n = M * K
        positions = positions.reshape(n, 3)
        scales = scales.reshape(n, 3)
        rotations = rotations.reshape(n, 4)
        opacities = opacities.reshape(n, 1)
        sh_dc = color.reshape(n, 3)
        sh_rest = torch.zeros(n, 45)

        if hard_filter:
            keep = torch.sigmoid(opacities.squeeze(-1)) > threshold
            positions, scales, rotations = positions[keep], scales[keep], rotations[keep]
            opacities, sh_dc, sh_rest = opacities[keep], sh_dc[keep], sh_rest[keep]

        return GaussianCloud(
            positions=positions, scales=scales, rotations=rotations,
            opacities=opacities, sh_dc=sh_dc, sh_rest=sh_rest,
        )


    mo.md("Defined `decode_to_gaussians()`.")

    return (decode_to_gaussians,)


@app.cell
def training_loop(
    CANONICAL_DISTANCE,
    F,
    LR,
    N_ITERS,
    anchors,
    camera_view_dirs,
    cameras,
    color_mlp,
    cov_mlp,
    decode_to_gaussians,
    mo,
    opacity_mlp,
    render,
    target_renders,
    torch,
):
    def train(anchors, opacity_mlp, color_mlp, cov_mlp, cameras, camera_view_dirs, target_renders, n_iters, lr):
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
        _g = torch.Generator().manual_seed(0)

        for it in range(n_iters):
            view_idx = torch.randint(0, n_views, (1,), generator=_g).item()
            cam = cameras[view_idx]
            view_dir = camera_view_dirs[view_idx]
            target_img = target_renders[view_idx]

            decoded = decode_to_gaussians(anchors, view_dir, CANONICAL_DISTANCE, hard_filter=False)
            pred_img = render(decoded, cam)
            loss = F.l1_loss(pred_img, target_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            if it % 100 == 0 or it == n_iters - 1:
                print(f"iter {it:5d}  loss {loss.item():.5f}")

        return loss_history


    loss_history = train(
        anchors, opacity_mlp, color_mlp, cov_mlp,
        cameras, camera_view_dirs, target_renders,
        N_ITERS, LR,
    )
    mo.md(f"Training done. Final loss: **{loss_history[-1]:.5f}** (started at {loss_history[0]:.5f}).")

    return (loss_history,)


@app.cell
def final_decode(
    CANONICAL_DISTANCE,
    K,
    OPACITY_THRESHOLD,
    anchors,
    decode_to_gaussians,
    mo,
    torch,
):
    CANONICAL_VIEW_DIR = torch.tensor([0.0, 0.0, 1.0])

    with torch.no_grad():
        reconstructed = decode_to_gaussians(
            anchors, CANONICAL_VIEW_DIR, CANONICAL_DISTANCE,
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
    F,
    K,
    active_gaussians,
    anchors,
    cameras,
    loss_history,
    mo,
    reconstructed,
    render,
    target_renders,
    torch,
):
    with torch.no_grad():
        _recon_renders = [render(reconstructed, cam) for cam in cameras]
        _l1_per_view = [F.l1_loss(r, t).item() for r, t in zip(_recon_renders, target_renders)]
        _l2_per_view = [F.mse_loss(r, t).item() for r, t in zip(_recon_renders, target_renders)]

    metrics = {
        "original Gaussians (N)": len(active_gaussians),
        "anchors (M)": len(anchors),
        "max candidates (M*K)": len(anchors) * K,
        "reconstructed Gaussians (N')": len(reconstructed),
        "mean photometric L1 across views": sum(_l1_per_view) / len(_l1_per_view),
        "mean photometric MSE across views": sum(_l2_per_view) / len(_l2_per_view),
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
    plt,
    reconstructed,
    render,
    target_renders,
    torch,
):
    _fig = plt.figure(figsize=(11, 6))

    _ax1 = _fig.add_subplot(2, 3, 1, projection="3d")
    _orig_pos = active_gaussians.positions.detach().numpy()
    _orig_col = torch.sigmoid(active_gaussians.sh_dc).detach().numpy()
    _ax1.scatter(_orig_pos[:, 0], _orig_pos[:, 1], _orig_pos[:, 2], c=_orig_col, s=4)
    _ax1.set_title("Original (standard 3DGS)")

    _ax2 = _fig.add_subplot(2, 3, 2, projection="3d")
    _recon_pos = reconstructed.positions.detach().numpy()
    _recon_col = torch.sigmoid(reconstructed.sh_dc).detach().numpy()
    _ax2.scatter(_recon_pos[:, 0], _recon_pos[:, 1], _recon_pos[:, 2], c=_recon_col, s=4)
    _ax2.set_title("Reconstructed (from anchors)")

    _ax3 = _fig.add_subplot(2, 3, 3, projection="3d")
    _anchor_pos = anchors.anchor_positions.detach().numpy()
    _ax3.scatter(_anchor_pos[:, 0], _anchor_pos[:, 1], _anchor_pos[:, 2], c="gray", s=8)
    _ax3.set_title(f"Anchors (M={len(anchors)})")

    _ax4 = _fig.add_subplot(2, 3, 4)
    _ax4.plot(loss_history)
    _ax4.set_title("Training loss (photometric L1)")
    _ax4.set_xlabel("iteration")

    _ax5 = _fig.add_subplot(2, 3, 5)
    _ax5.imshow(target_renders[0].detach().numpy())
    _ax5.set_title("Target render (view 0)")
    _ax5.axis("off")

    _ax6 = _fig.add_subplot(2, 3, 6)
    with torch.no_grad():
        _recon_view0 = render(reconstructed, cameras[0])
    _ax6.imshow(_recon_view0.detach().numpy())
    _ax6.set_title("Reconstructed render (view 0)")
    _ax6.axis("off")

    _fig.tight_layout()
    _fig

    return


if __name__ == "__main__":
    app.run()
