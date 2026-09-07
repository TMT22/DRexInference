"""File I/O utilities: mask loading, multi-modality loading, video/image saving, GPU logging."""

import numpy as np
import torch
import imageio
from PIL import Image
from typing import Dict, Optional, Tuple
import torchvision.transforms.functional as transforms_F

from cosmos_predict1.diffusion.inference.inference_utils import read_video_or_image_into_frames_BCTHW


# ---------------------------------------------------------------------------
# Mask loading
# ---------------------------------------------------------------------------

def _apply_pad_and_resize(
    input_tensor: np.ndarray,
    H: int,
    W: int,
    pad_with_zero: bool,
    rotate_90_clockwise: bool,
    nearest: bool,
) -> torch.Tensor:
    """Shared rotation → padding → resize pipeline for mask arrays (TCHW)."""
    if rotate_90_clockwise:
        input_tensor = np.rot90(input_tensor, k=-1, axes=(2, 3))
    if pad_with_zero:
        aspect = W / H
        t, c, h, w = input_tensor.shape
        target_w = int(h * aspect)
        if target_w > w:
            pad = target_w - w
            pad_l, pad_r = pad // 2, pad - pad // 2
            input_tensor = np.pad(
                input_tensor,
                ((0, 0), (0, 0), (0, 0), (pad_l, pad_r)),
                mode="constant",
                constant_values=0,
            )
    tensor = torch.from_numpy(input_tensor)
    if H is not None and W is not None:
        mode = transforms_F.InterpolationMode.NEAREST if nearest else transforms_F.InterpolationMode.BILINEAR
        tensor = transforms_F.resize(tensor, size=(H, W), interpolation=mode, antialias=not nearest)
        tensor = (tensor > 0).float()
    return tensor


def load_mask_from_png(
    filepath: str,
    rotate_90_clockwise: bool = False,
    pad_with_zero: bool = True,
    H: int = 704,
    W: int = 1280,
    num_frames: int = -1,
) -> torch.Tensor:
    """Load a mask from a PNG with an alpha channel. Returns BCTHW tensor."""
    img = Image.open(filepath).convert("RGBA")
    alpha = np.array(img)[..., 3]
    mask = (alpha > 0).astype(np.float32)[None, None]  # (1, 1, H, W)

    tensor = _apply_pad_and_resize(mask, H, W, pad_with_zero, rotate_90_clockwise, nearest=True)
    if num_frames > 0:
        tensor = tensor.repeat(num_frames, 1, 1, 1)
    # TCHW → BCTHW
    return tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()


def load_mask(
    filepath: str,
    rotate_90_clockwise: bool = False,
    pad_with_zero: bool = True,
    H: int = 704,
    W: int = 1280,
    num_frames: int = -1,
) -> torch.Tensor:
    """Load a mask from a single-channel greyscale PNG. Returns BCTHW tensor."""
    img = Image.open(filepath).convert("RGB")
    raw = np.array(img)[..., 0].astype(np.float32) / 255.0
    mask = (raw > 0.5).astype(np.float32)[None, None]  # (1, 1, H, W)

    tensor = _apply_pad_and_resize(mask, H, W, pad_with_zero, rotate_90_clockwise, nearest=True)
    if num_frames > 0:
        tensor = tensor.repeat(num_frames, 1, 1, 1)
    return tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()


# ---------------------------------------------------------------------------
# Multi-modality loading
# ---------------------------------------------------------------------------

def load_modalities(
    modal_dict: Dict[str, str],
    rotate_90_clockwise: bool = False,
    pad_with_zero: bool = True,
    H: int = 704,
    W: int = 1280,
    num_frames: int = -1,
) -> Dict[str, torch.Tensor]:
    """Load multiple image/video paths into BCTHW tensors.

    Parameters
    ----------
    modal_dict : {name: filepath}
    rotate_90_clockwise, pad_with_zero, H, W : forwarded to the frame reader.
    num_frames : if > 0, clips (or pads by repeating) to exactly this many frames.

    Returns
    -------
    {name: BCTHW tensor}
    """
    loaded: Dict[str, torch.Tensor] = {}
    for key, path in modal_dict.items():
        t = read_video_or_image_into_frames_BCTHW(
            path,
            rotate_90_clockwise=rotate_90_clockwise,
            pad_with_zero=pad_with_zero,
            H=H,
            W=W,
        )
        if num_frames > 0:
            t = t[:, :, :num_frames]
            if t.shape[2] < num_frames:
                pad = t[:, :, -1:].repeat(1, 1, num_frames - t.shape[2], 1, 1)
                t = torch.cat([t, pad], dim=2).clone()
        loaded[key] = t
    return loaded


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_video_tensor_to_mp4(
    tensor: torch.Tensor,
    output_path: str,
    undo_rotation: bool = False,
    fps: float = 24.0,
) -> None:
    """Write a BCTHW tensor (values in [0, 1]) to an MP4 file.

    Only the first batch element is written.
    """
    vid = tensor[0]  # (C, T, H, W)
    C, T, H, W = vid.shape
    writer = imageio.get_writer(output_path, fps=fps)
    for t in range(T):
        frame = vid[:, t].permute(1, 2, 0).detach().to(torch.float32).cpu().contiguous().numpy()
        if undo_rotation:
            frame = np.rot90(frame, k=1)
        writer.append_data(np.clip(frame * 255, 0, 255).astype(np.uint8))
    writer.close()


def save_image_tensor_to_jpg(
    tensor: torch.Tensor,
    output_path: str,
    undo_rotation: bool = False,
    frame_idx: int = 0,
) -> None:
    """Write a single frame from a BCTHW tensor (values in [0, 1]) to a JPEG file."""
    img = tensor[0, :, frame_idx]  # (C, H, W)
    frame = img.permute(1, 2, 0).detach().to(torch.float32).cpu().contiguous().numpy()
    if undo_rotation:
        frame = np.rot90(frame, k=1)
    imageio.imwrite(output_path, np.clip(frame * 255, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_gpu_stats(prefix: str = "", device: int = 0) -> None:
    """Print GPU memory and utilisation for *device* to stdout.

    Uses ``torch.cuda`` for memory and ``pynvml`` for utilisation (SM
    occupancy).  If ``pynvml`` is not available, the utilisation column is
    omitted gracefully.

    Parameters
    ----------
    prefix : str
        Short label prepended to the output line (e.g. ``"iter 10"``).
    device : int
        CUDA device index.

    Example output
    --------------
    [iter 10] GPU 0  alloc=3.41 GB  reserved=4.00 GB  free=11.23 GB  util=47%
    """
    if not torch.cuda.is_available():
        print(f"{('[' + prefix + '] ') if prefix else ''}GPU stats unavailable (no CUDA)", flush=True)
        return

    alloc    = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    total    = torch.cuda.get_device_properties(device).total_memory
    free     = total - reserved

    def _gb(b: int) -> str:
        return f"{b / 1024 ** 3:.2f} GB"

    util_str = ""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle   = pynvml.nvmlDeviceGetHandleByIndex(device)
        rates    = pynvml.nvmlDeviceGetUtilizationRates(handle)
        util_str = f"  util={rates.gpu}%  mem_util={rates.memory}%"
    except Exception:
        pass  # pynvml absent or query failed — skip silently

    tag = f"[{prefix}] " if prefix else ""
    print(
        f"{tag}GPU {device}  "
        f"alloc={_gb(alloc)}  reserved={_gb(reserved)}  free={_gb(free)}"
        f"{util_str}",
        flush=True,
    )
