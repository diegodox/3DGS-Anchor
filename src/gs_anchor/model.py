import torch
import torch.nn as nn

from .gaussians import AnchorCloud, GaussianCloud


class OpacityMLP(nn.Module):
    def __init__(self, feature_dim: int, k: int, hidden: int = 64):
        super().__init__()
        input_dim = feature_dim + 3 + 1  # anchor feature + view direction + distance
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, k),
        )

    def forward(self, x):
        return self.net(x)  # raw logits, [.., K]


class ColorMLP(nn.Module):
    def __init__(self, feature_dim: int, k: int, hidden: int = 64):
        super().__init__()
        input_dim = feature_dim + 3 + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, k * 3),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))  # RGB in [0,1], [.., K*3]


class CovMLP(nn.Module):
    def __init__(self, feature_dim: int, k: int, hidden: int = 64):
        super().__init__()
        input_dim = feature_dim + 3 + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, k * 7),
        )

    def forward(self, x):
        return self.net(x)  # [.., K*7]: 3 scale-offset + 4 quat-offset per neighbor


def decode_to_gaussians(
    anchors: AnchorCloud,
    opacity_mlp: OpacityMLP,
    color_mlp: ColorMLP,
    cov_mlp: CovMLP,
    view_dir: torch.Tensor,
    distance: float,
    hard_filter: bool = False,
    threshold: float = 0.005,
) -> GaussianCloud:
    """Decode anchors+MLPs into a flat GaussianCloud for a given viewing
    direction/distance. `hard_filter=False` (training mode) keeps all M*K
    soft-opacity candidates so the photometric loss can shape them via
    compositing. `hard_filter=True` (final structured->standard decode)
    drops near-zero-opacity candidates, matching the paper's pruning."""
    M = len(anchors)
    k = anchors.anchor_offsets.shape[1]
    scaling = anchors.anchor_scaling.clamp_min(1e-4)
    view_dir = view_dir / view_dir.norm()
    view_dir_b = view_dir.view(1, 3).expand(M, 3)
    distance_b = torch.full((M, 1), float(distance))
    mlp_in = torch.cat([anchors.anchor_features, view_dir_b, distance_b], dim=-1)

    opacity_logits = opacity_mlp(mlp_in)               # [M,K]
    color = color_mlp(mlp_in).view(M, k, 3)             # [M,K,3]
    cov_out = cov_mlp(mlp_in).view(M, k, 7)             # [M,K,7]
    scale_offset = cov_out[..., :3]
    quat_offset = cov_out[..., 3:7]

    positions = anchors.anchor_positions.unsqueeze(1) + anchors.anchor_offsets * scaling.unsqueeze(1)
    scales = torch.log(scaling).unsqueeze(1) + scale_offset
    rotations = quat_offset
    opacities = opacity_logits.unsqueeze(-1)

    n = M * k
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
