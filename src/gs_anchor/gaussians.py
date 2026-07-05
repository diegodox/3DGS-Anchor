from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GaussianCloud:
    positions: torch.Tensor   # [N,3] float32
    scales: torch.Tensor      # [N,3] log-scale
    rotations: torch.Tensor   # [N,4] quaternion (unnormalized)
    opacities: torch.Tensor   # [N,1] logit
    sh_dc: torch.Tensor       # [N,3] RGB-ready color (degree-0 SH already converted)
    sh_rest: torch.Tensor     # [N,45] higher-order SH (zero-filled if unused)

    def __len__(self):
        return self.positions.shape[0]

    def to(self, device):
        return GaussianCloud(
            positions=self.positions.to(device),
            scales=self.scales.to(device),
            rotations=self.rotations.to(device),
            opacities=self.opacities.to(device),
            sh_dc=self.sh_dc.to(device),
            sh_rest=self.sh_rest.to(device),
        )


@dataclass
class AnchorCloud:
    anchor_positions: torch.Tensor  # [M,3] fixed
    anchor_features: torch.Tensor   # [M,F] nn.Parameter
    anchor_scaling: torch.Tensor    # [M,3] nn.Parameter
    anchor_offsets: torch.Tensor    # [M,K,3] nn.Parameter

    def __len__(self):
        return self.anchor_positions.shape[0]

    def to(self, device):
        """Move to device. Only safe to call before constructing an
        optimizer, since the nn.Parameter fields are re-wrapped."""
        return AnchorCloud(
            anchor_positions=self.anchor_positions.to(device),
            anchor_features=nn.Parameter(self.anchor_features.to(device)),
            anchor_scaling=nn.Parameter(self.anchor_scaling.to(device)),
            anchor_offsets=nn.Parameter(self.anchor_offsets.to(device)),
        )


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


def make_synthetic_gaussians(n: int = 800, seed: int = 0) -> GaussianCloud:
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


def subsample_gaussians(gaussians: GaussianCloud, n: int, mode: str = "topk_opacity", seed: int = 0) -> GaussianCloud:
    """Reduce a GaussianCloud to at most n points -- real scenes are far too
    large for this pipeline's dense, non-tiled renderer. `topk_opacity` keeps
    the most visually significant points; `random` is a seeded uniform sample."""
    total = len(gaussians)
    if total <= n:
        return gaussians

    if mode == "topk_opacity":
        idx = torch.argsort(gaussians.opacities.squeeze(-1), descending=True)[:n]
    elif mode == "random":
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(total, generator=g)[:n]
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    return GaussianCloud(
        positions=gaussians.positions[idx],
        scales=gaussians.scales[idx],
        rotations=gaussians.rotations[idx],
        opacities=gaussians.opacities[idx],
        sh_dc=gaussians.sh_dc[idx],
        sh_rest=gaussians.sh_rest[idx],
    )


def crop_to_density_core(
    gaussians: GaussianCloud, voxel_size: float = 2.0, quantile: float = 0.9995, margin: float = 2.0
) -> GaussianCloud:
    """Crop to the densest contiguous region, found via voxel occupancy
    counts. Real-world captures (e.g. a full street-level COLMAP scan) mix a
    dense object of interest with a sparse, far-flung background/
    environment; the object's local point density is far higher than the
    background's, so thresholding voxel counts isolates it with no semantic
    labels needed. Confirmed on the Tanks & Temples "truck" scan: the whole
    capture spans ~180 units across an intersection, but the truck itself is
    a ~4-unit-radius dense cluster -- a whole-cloud centroid/std treats the
    sparse street network as part of the "object," which is why cameras and
    subsampled points landed mostly off the truck entirely."""
    positions = gaussians.positions
    mins = positions.min(dim=0).values
    idx = torch.floor((positions - mins) / voxel_size).long()
    key = idx[:, 0] * 1_000_000 + idx[:, 1] * 1_000 + idx[:, 2]
    _, inverse, counts = torch.unique(key, return_inverse=True, return_counts=True)

    thresh = torch.quantile(counts.float(), quantile)
    core_mask = (counts > thresh)[inverse]
    core = positions[core_mask]
    centroid = core.mean(dim=0)
    radius = (core - centroid).norm(dim=-1).quantile(0.95) * margin

    roi_idx = ((positions - centroid).norm(dim=-1) < radius).nonzero(as_tuple=True)[0]
    return GaussianCloud(
        positions=gaussians.positions[roi_idx],
        scales=gaussians.scales[roi_idx],
        rotations=gaussians.rotations[roi_idx],
        opacities=gaussians.opacities[roi_idx],
        sh_dc=gaussians.sh_dc[roi_idx],
        sh_rest=gaussians.sh_rest[roi_idx],
    )
