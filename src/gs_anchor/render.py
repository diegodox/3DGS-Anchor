"""Rendering via gsplat (https://github.com/nerfstudio-project/gsplat) --
real tile-based CUDA rasterization with full anisotropic 3D covariance
(uses `rotations`, not a simplified isotropic approximation).

gsplat JIT-compiles its CUDA extension on first import. In a sandbox that
assembles CUDA from pip `nvidia-*` packages rather than a system CUDA
toolkit install, `CUDA_HOME`/`PATH` must point at the `nvidia/cu*` package
directory *before* gsplat is imported -- `_ensure_cuda_home()` below does
this automatically by locating the installed `nvidia` package's directory
and finding `cu*/bin/nvcc` inside it. No manual environment setup needed.

Once compiled, both forward and backward (training) work correctly,
confirmed on an NVIDIA RTX PRO 6000 Blackwell (compute capability 12.0).
"""

import math
import os
import glob
import importlib.util


def _ensure_cuda_home() -> None:
    if os.environ.get("CUDA_HOME"):
        return
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return
    for base in spec.submodule_search_locations:
        for cu_dir in glob.glob(os.path.join(base, "cu*")):
            nvcc = os.path.join(cu_dir, "bin", "nvcc")
            if os.path.exists(nvcc):
                os.environ["CUDA_HOME"] = cu_dir
                os.environ["PATH"] = os.path.join(cu_dir, "bin") + os.pathsep + os.environ.get("PATH", "")
                return


_ensure_cuda_home()

import torch
import gsplat

from .gaussians import Camera, GaussianCloud


def render(gaussians: GaussianCloud, camera: Camera) -> torch.Tensor:
    """Render an [H,W,3] image via gsplat's tile-based CUDA rasterizer."""
    device = gaussians.positions.device

    means = gaussians.positions
    quats = gaussians.rotations / gaussians.rotations.norm(dim=-1, keepdim=True)
    scales = torch.exp(gaussians.scales)
    opacities = torch.sigmoid(gaussians.opacities.squeeze(-1))
    colors = gaussians.sh_dc.clamp(0.0, 1.0)

    viewmat = torch.zeros(4, 4, device=device)
    viewmat[:3, :3] = camera.R
    viewmat[:3, 3] = camera.t
    viewmat[3, 3] = 1.0

    K = torch.zeros(3, 3, device=device)
    K[0, 0] = camera.focal
    K[1, 1] = camera.focal
    K[0, 2] = camera.principal[0]
    K[1, 2] = camera.principal[1]
    K[2, 2] = 1.0

    render_colors, _render_alphas, _meta = gsplat.rasterization(
        means, quats, scales, opacities, colors,
        viewmat[None], K[None],
        width=camera.W, height=camera.H,
        sh_degree=None,
    )
    return render_colors[0].clamp(0.0, 1.0)


def look_at_camera(eye: torch.Tensor, target: torch.Tensor, focal: float, size: int) -> Camera:
    device = eye.device
    forward = (target - eye)
    forward = forward / forward.norm()
    up_hint = torch.tensor([0.0, 1.0, 0.0], device=device)
    if torch.abs(torch.dot(forward, up_hint)) > 0.99:
        up_hint = torch.tensor([1.0, 0.0, 0.0], device=device)
    right = torch.linalg.cross(forward, up_hint)
    right = right / right.norm()
    up = torch.linalg.cross(right, forward)
    # world->camera rotation: rows are camera axes (right, up, -forward... use +forward as +z)
    R = torch.stack([right, up, forward], dim=0)
    t = -R @ eye
    return Camera(R=R, t=t, focal=focal, H=size, W=size)


def make_synthetic_cameras(centroid: torch.Tensor, radius: float, n_views: int, size: int):
    device = centroid.device
    cams = []
    dirs_list = []
    golden_angle = math.pi * (3.0 - 5.0 ** 0.5)
    for i in range(n_views):
        yy = 1 - 2 * (i / max(n_views - 1, 1))
        r = (1 - yy * yy) ** 0.5
        theta = golden_angle * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        d = torch.tensor([x, yy, z], device=device)
        eye = centroid + radius * d
        cams.append(look_at_camera(eye, centroid, focal=1.2 * size, size=size))
        dirs_list.append(-d)
    return cams, torch.stack(dirs_list)
