"""Camera calibration I/O and coordinate-system conversion utilities."""

import os
import numpy as np
import torch
from typing import List, Optional


def read_calib(
    calib_dir,
    IMAGE_W: int,
    INTR: list,
    EXTR: list,
    scale_to_meters: bool = False,
    DISTOR: Optional[list] = None,
    invert_extr: bool = True,
    focal_length: Optional[list] = None,
):
    """Read camera intrinsics, extrinsics, and distortion from a .calib file.

    Parameters
    ----------
    calib_dir : str or Path
        Directory containing cameras.calib (or camera.calib).
    IMAGE_W : int
        Image width used to un-normalise intrinsics. Pass any dummy value when
        only extrinsics are needed.
    INTR : list
        Empty list; populated in-place with (4×4) intrinsic matrices.
    EXTR : list
        Empty list; populated in-place with (4×4) extrinsic matrices
        (c2w when invert_extr=True, w2c otherwise).
    scale_to_meters : bool
        Divide translation by 1000 (mm → m).
    DISTOR : list or None
        If provided, populated with distortion coefficient arrays.
    invert_extr : bool
        Return c2w matrices (True) instead of w2c (False).
    focal_length : list or None
        If provided, populated with scalar focal lengths.
    """
    calib_dir = str(calib_dir)
    INTR.clear()
    EXTR.clear()
    if DISTOR is not None:
        DISTOR.clear()
    if focal_length is not None:
        focal_length.clear()

    calib_path = os.path.join(calib_dir, "cameras.calib")
    if not os.path.isfile(calib_path):
        calib_path = calib_path.replace("cameras.calib", "camera.calib")

    with open(calib_path, "r") as fp:
        while True:
            text = fp.readline()
            if not text:
                break
            if "distortionModel" in text:
                continue
            if "distortion" in text and DISTOR is not None:
                dist = np.fromstring(text.replace("distortion", ""), dtype=float, sep=" ")
                DISTOR.append(dist)
            if "focalLength" in text and focal_length is not None:
                focal_length.append(float(text.replace("focalLength", "").split()[0]))
            if "pixelAspect" in text:
                pass  # read but unused
            elif "extrinsic" in text:
                lines = []
                for _ in range(3):
                    line = fp.readline()
                    lines.append([float(w) for w in line.strip("#").strip().split()])
                lines.append([0, 0, 0, 1])
                extrinsic = np.array(lines, dtype=np.float32)
                if scale_to_meters:
                    extrinsic[:3, 3] /= 1000.0
                EXTR.append(np.linalg.inv(extrinsic) if invert_extr else extrinsic)
            elif "intrinsic" in text:
                lines = []
                line = fp.readline()
                if "time" in line:
                    continue
                for _ in range(3):
                    row = [float(w) for w in line.strip("#").strip().split()]
                    row.append(0)
                    lines.append(row)
                    line = fp.readline()
                lines.append([0, 0, 0, 1])
                intrinsic = np.array(lines, dtype=np.float32)
                intrinsic[0, 2] *= IMAGE_W
                intrinsic[1, 2] *= IMAGE_W
                intrinsic[0, 0] *= IMAGE_W
                intrinsic[1, 1] *= IMAGE_W
                INTR.append(intrinsic)


def convert_c2w_opengl_to_opencv(c2w_opengl: np.ndarray, with_rotation: bool = False) -> np.ndarray:
    """Convert an OpenGL-convention c2w matrix to OpenCV convention.

    Optionally applies a 90° clockwise camera roll (needed for portrait captures).

    Parameters
    ----------
    c2w_opengl : np.ndarray, shape (4, 4)
    with_rotation : bool
        Apply a 90° CW roll around the camera forward axis.

    Returns
    -------
    np.ndarray, shape (4, 4)
    """
    flip_z = np.diag([1.0, 1.0, -1.0, 1.0])
    c2w_cv = c2w_opengl @ flip_z

    R = c2w_cv[:3, :3]
    C = c2w_cv[:3, 3]

    if with_rotation:
        R_90cw = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
        R = R @ R_90cw.T

    c2w_final = np.eye(4)
    c2w_final[:3, :3] = R
    c2w_final[:3, 3] = C
    return c2w_final


def world_dirs_to_camera_dirs(normals_world: torch.Tensor, c2w_cv: np.ndarray) -> torch.Tensor:
    """Rotate world-space direction vectors into camera space.

    Parameters
    ----------
    normals_world : torch.Tensor, shape (H, W, 3) or (B, H, W, 3)
    c2w_cv : np.ndarray, shape (4, 4)
        OpenCV camera-to-world matrix.

    Returns
    -------
    torch.Tensor  same shape as normals_world, L2-normalised.
    """
    R_c2w = torch.tensor(c2w_cv[:3, :3], dtype=normals_world.dtype, device=normals_world.device)
    R_w2c = R_c2w.T  # orthonormal → transpose == inverse

    is_batched = normals_world.dim() == 4
    if not is_batched:
        normals_world = normals_world.unsqueeze(0)

    B, H, W, _ = normals_world.shape
    dirs = normals_world.reshape(B, -1, 3)
    dirs_cam = torch.matmul(dirs, R_w2c.T)
    normals_cam = dirs_cam.reshape(B, H, W, 3)

    if not is_batched:
        normals_cam = normals_cam.squeeze(0)

    return torch.nn.functional.normalize(normals_cam, dim=-1)
