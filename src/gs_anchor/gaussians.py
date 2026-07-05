from dataclasses import dataclass

import torch


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
    anchor_features: torch.Tensor   # [M,F] nn.Parameter
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
