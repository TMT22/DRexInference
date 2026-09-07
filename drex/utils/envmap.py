"""Environment-map loading and camera-space projection utilities."""

import torch
from torchvision import transforms
from PIL import Image
from typing import List, Tuple

from drex.utils.camera import convert_c2w_opengl_to_opencv, world_dirs_to_camera_dirs


def load_env_nrm(
    env_nrm_path: str,
    flip_normals: bool = True,
    resolution: Tuple[int, int] = (704, 1280),
    device: torch.device = None,
) -> torch.Tensor:
    """Load environment-map normal directions from an image file.

    Parameters
    ----------
    env_nrm_path : str
        Path to the environment normals image (RGB encoded directions).
    flip_normals : bool
        Multiply by -1 after normalising to [-1, 1].
    resolution : (H, W)
        Target resolution.
    device : torch.device or None

    Returns
    -------
    torch.Tensor, shape (H, W, 3), values in [-1, 1].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR,
                          max_size=None, antialias=True),
    ])
    raw = transform(Image.open(env_nrm_path))
    env_nrm = raw.permute(1, 2, 0).float().to(device) / 127.5 - 1.0  # (H, W, 3)
    if flip_normals:
        env_nrm *= -1.0
    return env_nrm


def prepare_env_nrms(
    env_nrm: torch.Tensor,
    opencv_c2w_list: list,
    apply_rotation: bool = False,
    num_frames: int = 1,
    rotate_envlight: bool = False,
) -> torch.Tensor:
    """Project world-space envmap normals into camera space for each frame.

    Parameters
    ----------
    env_nrm : torch.Tensor, shape (H, W, 3)
        World-space environment normals.
    opencv_c2w_list : list of np.ndarray (4×4)
        Per-frame camera-to-world matrices.  If shorter than num_frames, the
        last entry is repeated.
    apply_rotation : bool
        Pass with_rotation=True to convert_c2w_opengl_to_opencv.
    num_frames : int
    rotate_envlight : bool
        Progressively shift the envmap horizontally across frames.

    Returns
    -------
    torch.Tensor, shape (1, 3, T, H, W)
    """
    if len(opencv_c2w_list) < num_frames:
        print(
            f"Warning: only {len(opencv_c2w_list)} c2w matrices for {num_frames} frames; "
            "repeating the last one."
        )

    rotation_pixel_shift = env_nrm.shape[1] // num_frames if rotate_envlight else 0

    frames = []
    for i in range(num_frames):
        c2w = convert_c2w_opengl_to_opencv(
            opencv_c2w_list[min(i, len(opencv_c2w_list) - 1)],
            with_rotation=apply_rotation,
        )
        frame = world_dirs_to_camera_dirs(env_nrm, c2w).clone()
        if rotate_envlight:
            frame = torch.roll(frame, shifts=i * rotation_pixel_shift, dims=1)
        frames.append(frame)

    # (T, H, W, 3) → (1, 3, T, H, W)
    return torch.stack(frames, dim=0).unsqueeze(0).permute(0, 4, 1, 2, 3)
