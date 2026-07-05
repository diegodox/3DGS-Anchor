"""Read a COLMAP sparse reconstruction (cameras.bin / images.bin) and turn
its known, real training-camera poses into this project's `Camera`s --
used instead of guessing camera placement (random or density-heuristic)
for scenes whose original calibration is available, since those poses are
guaranteed to frame the captured subject correctly (that's what they were
pointed at when the photos were taken).

Binary layout follows COLMAP's own format (see COLMAP's `read_write_model.py`
for the reference implementation this mirrors).
"""

import os
import struct

import numpy as np
import torch

from .gaussians import Camera

_CAMERA_MODEL_NUM_PARAMS = {
    0: 3,   # SIMPLE_PINHOLE: f, cx, cy
    1: 4,   # PINHOLE: fx, fy, cx, cy
    2: 4,   # SIMPLE_RADIAL: f, cx, cy, k
    3: 5,   # RADIAL
    4: 8,   # OPENCV
    5: 8,   # OPENCV_FISHEYE
    6: 12,  # FULL_OPENCV
    7: 5,   # FOV
    8: 4,   # SIMPLE_RADIAL_FISHEYE
    9: 5,   # RADIAL_FISHEYE
    10: 12,  # THIN_PRISM_FISHEYE
}


def _read(fid, num_bytes, fmt):
    return struct.unpack("<" + fmt, fid.read(num_bytes))


def read_cameras_binary(path: str) -> dict:
    """-> {camera_id: (width, height, fx, fy, cx, cy)}. Non-focal distortion
    params (radial/tangential) are dropped -- we only need fx/fy/cx/cy to
    place a pinhole camera; COLMAP's SIMPLE_* models share (f, cx, cy) for
    both axes."""
    cameras = {}
    with open(path, "rb") as fid:
        (num_cameras,) = _read(fid, 8, "Q")
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read(fid, 24, "iiQQ")
            num_params = _CAMERA_MODEL_NUM_PARAMS[model_id]
            params = _read(fid, 8 * num_params, "d" * num_params)
            if num_params >= 4:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            else:
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            cameras[camera_id] = (width, height, fx, fy, cx, cy)
    return cameras


def read_images_binary(path: str) -> list:
    """-> [(qvec[4] wxyz, tvec[3], camera_id), ...]."""
    images = []
    with open(path, "rb") as fid:
        (num_images,) = _read(fid, 8, "Q")
        for _ in range(num_images):
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = _read(fid, 64, "idddddddi")
            while _read(fid, 1, "c")[0] != b"\x00":
                pass
            (num_points2d,) = _read(fid, 8, "Q")
            fid.read(24 * num_points2d)
            images.append(((qw, qx, qy, qz), (tx, ty, tz), camera_id))
    return images


def qvec2rotmat(qvec) -> np.ndarray:
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def load_colmap_cameras(sparse_dir: str, render_size: int):
    """Load every registered training view in a COLMAP `sparse/0` directory
    as a `Camera` scaled to `render_size`x`render_size` (this pipeline
    always renders square). Returns (cameras, view_dirs[N,3]) -- view_dirs
    is each camera's forward direction in world space (camera->world
    rotation applied to the camera-space forward axis), matching the
    "direction from eye into the scene" convention used elsewhere in this
    project (e.g. `make_random_cameras`)."""
    intrinsics = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    extrinsics = read_images_binary(os.path.join(sparse_dir, "images.bin"))

    cameras = []
    view_dirs = []
    for qvec, tvec, camera_id in extrinsics:
        width, height, fx, fy, cx, cy = intrinsics[camera_id]
        R = torch.tensor(qvec2rotmat(qvec), dtype=torch.float32)
        t = torch.tensor(tvec, dtype=torch.float32)

        scale = render_size / max(width, height)
        focal = float(fx) * scale

        cameras.append(Camera(R=R, t=t, focal=focal, H=render_size, W=render_size))
        view_dirs.append(R.T @ torch.tensor([0.0, 0.0, 1.0]))

    return cameras, torch.stack(view_dirs)
