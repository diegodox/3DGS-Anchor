import torch

from .gaussians import Camera, GaussianCloud


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
