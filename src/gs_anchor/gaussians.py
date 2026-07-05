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
