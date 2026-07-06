import torch
import torch.nn as nn

from .gaussians import AnchorCloud, GaussianCloud


def _count_voxels(positions: torch.Tensor, voxel_size: float) -> int:
    voxel_idx = torch.floor(positions / voxel_size).to(torch.int64)
    return torch.unique(voxel_idx, dim=0).shape[0]


def choose_voxel_size(positions: torch.Tensor, k: int, target_ratio: float = 1.0) -> float:
    """Bisect a voxel size so `M*k` (anchor slot-capacity) lands near
    `target_ratio * len(positions)` -- i.e. give the anchor representation
    about as much raw capacity as the source point count, instead of a
    fixed voxel size that doesn't adapt to scene extent. A fixed voxel
    size is the wrong knob for real-world captures that mix a dense
    subject with a sparse, far-flung background (confirmed on the Tanks &
    Temples "truck" scan: `voxel_size=2.0` on the 207-unit-wide scene put
    96%+ of anchors in the background, leaving the actual photographed
    subject with ~50 anchors) -- counting unique voxels is cheap (a few ms
    regardless of size), so bisection is effectively free compared to the
    per-anchor warm-start loop in `build_anchors`."""
    n = positions.shape[0]
    target_m = max(1, int(target_ratio * n / k))

    extent = (positions.max(dim=0).values - positions.min(dim=0).values).norm().item()
    voxel_size = max(extent / (target_m ** (1 / 3)), 1e-6)

    lo = hi = voxel_size
    for _ in range(60):
        if _count_voxels(positions, lo) >= target_m:
            break
        lo /= 2.0
    for _ in range(60):
        if _count_voxels(positions, hi) <= target_m:
            break
        hi *= 2.0

    for _ in range(30):
        mid = (lo * hi) ** 0.5
        if _count_voxels(positions, mid) > target_m:
            lo = mid
        else:
            hi = mid

    return (lo * hi) ** 0.5


def build_anchors(gaussians: GaussianCloud, voxel_size: float, k: int, feature_dim: int):
    """Voxelize Gaussian positions into anchors; warm-start scaling/offsets
    from the Gaussians assigned to each anchor. These assignments are used
    only for initialization, not as training targets (training is
    render-based, see train.py)."""
    device = gaussians.positions.device
    positions = gaussians.positions
    voxel_idx = torch.floor(positions / voxel_size).to(torch.int64)
    unique_idx, inverse = torch.unique(voxel_idx, dim=0, return_inverse=True)
    anchor_positions = unique_idx.float() * voxel_size + voxel_size / 2
    M = anchor_positions.shape[0]

    order = torch.argsort(inverse)
    counts = torch.bincount(inverse, minlength=M)
    starts = torch.cumsum(counts, dim=0) - counts

    anchor_offsets_init = torch.zeros(M, k, 3, device=device)
    anchor_scaling_init = torch.full((M, 3), voxel_size / 2, device=device)
    coverage_mask = torch.zeros(M, k, dtype=torch.bool, device=device)

    rng = torch.Generator().manual_seed(0)
    for m in range(M):
        start = int(starts[m])
        count = int(counts[m])
        idxs = order[start:start + count]
        if count > k:
            perm = torch.randperm(count, generator=rng)[:k]
            idxs = idxs[perm.to(device)]
        n_assigned = idxs.shape[0]
        if n_assigned == 0:
            continue
        assigned_pos = positions[idxs]
        spread = assigned_pos.std(dim=0, unbiased=False).clamp_min(voxel_size * 0.1)
        anchor_scaling_init[m] = spread
        anchor_offsets_init[m, :n_assigned] = (assigned_pos - anchor_positions[m]) / spread
        coverage_mask[m, :n_assigned] = True

    anchor_features = nn.Parameter(0.01 * torch.randn(M, feature_dim, device=device))
    anchor_scaling = nn.Parameter(anchor_scaling_init.clone())
    anchor_offsets = nn.Parameter(anchor_offsets_init + 0.01 * torch.randn(M, k, 3, device=device))

    anchors = AnchorCloud(
        anchor_positions=anchor_positions,
        anchor_features=anchor_features,
        anchor_scaling=anchor_scaling,
        anchor_offsets=anchor_offsets,
    )
    return anchors, coverage_mask
