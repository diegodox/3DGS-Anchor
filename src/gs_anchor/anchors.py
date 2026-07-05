import torch
import torch.nn as nn

from .gaussians import AnchorCloud, GaussianCloud


def build_anchors(gaussians: GaussianCloud, voxel_size: float, k: int, feature_dim: int):
    """Voxelize Gaussian positions into anchors; warm-start scaling/offsets
    from the Gaussians assigned to each anchor. These assignments are used
    only for initialization, not as training targets (training is
    render-based, see train.py)."""
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

    rng = torch.Generator().manual_seed(0)
    for m in range(M):
        start = int(starts[m])
        count = int(counts[m])
        idxs = order[start:start + count]
        if count > k:
            perm = torch.randperm(count, generator=rng)[:k]
            idxs = idxs[perm]
        n_assigned = idxs.shape[0]
        if n_assigned == 0:
            continue
        assigned_pos = positions[idxs]
        spread = assigned_pos.std(dim=0, unbiased=False).clamp_min(voxel_size * 0.1)
        anchor_scaling_init[m] = spread
        anchor_offsets_init[m, :n_assigned] = (assigned_pos - anchor_positions[m]) / spread
        coverage_mask[m, :n_assigned] = True

    anchor_features = nn.Parameter(0.01 * torch.randn(M, feature_dim))
    anchor_scaling = nn.Parameter(anchor_scaling_init.clone())
    anchor_offsets = nn.Parameter(anchor_offsets_init + 0.01 * torch.randn(M, k, 3))

    anchors = AnchorCloud(
        anchor_positions=anchor_positions,
        anchor_features=anchor_features,
        anchor_scaling=anchor_scaling,
        anchor_offsets=anchor_offsets,
    )
    return anchors, coverage_mask
