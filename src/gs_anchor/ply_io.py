import numpy as np
import torch
from plyfile import PlyData

from .gaussians import GaussianCloud

SH_C0 = 0.28209479177387814  # degree-0 spherical harmonics basis constant


def load_ply_gaussians(path: str, load_sh_rest: bool = False) -> GaussianCloud:
    """Load a standard trained-3DGS .ply into a GaussianCloud.

    `f_dc_0/1/2` are raw degree-0 SH coefficients, converted here to
    RGB-ready color (`SH_C0 * f_dc + 0.5`) so `sh_dc` means the same thing
    everywhere in this pipeline (synthetic, real-ply, and MLP-decoded).

    `load_sh_rest` defaults to False since this pipeline never reconstructs
    higher-order SH -- skipping it is a large load-time/memory win on real
    scenes with millions of points.
    """
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
    f_dc = torch.tensor(
        np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32)
    )
    sh_dc = SH_C0 * f_dc + 0.5

    rest_names = sorted(
        (n for n in names if n.startswith("f_rest_")),
        key=lambda n: int(n.split("_")[-1]),
    )
    if load_sh_rest and rest_names:
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
