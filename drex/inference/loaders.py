"""Batched data loaders for inference and relighting pipelines.

These loaders produce data-batch dicts (with B=1) ready for
``drex.utils.batch.run_custom_diffrender_model``.

Classes
-------
BatchedImageDirLoader
    Loads image sequences from a directory in overlapping temporal chunks.
BatchedEnvLoaderSingle
    Loads a single HDR env map, optionally applying a per-frame camera
    trajectory from a JSON file or calibration directory.
BatchedEnvLoaderDataset
    Loads per-frame env maps stored in an ERA-style dataset tree.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.utils_env_proj import (
    process_environment_map,
)
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.rendering_utils import envmap_vec

from drex.utils.camera import read_calib, convert_c2w_opengl_to_opencv
from drex.utils.batch import prepare_base_data_batch, concatenate_data_batches
from drex.utils.io import load_modalities


# ---------------------------------------------------------------------------
# Image sequence loader
# ---------------------------------------------------------------------------

class BatchedImageDirLoader:
    """Load an image directory in overlapping temporal chunks.

    Each call to ``next_chunk()`` returns a data-batch dict covering
    ``T_batch_size`` consecutive frames.  The pointer advances by
    ``T_batch_size - overlap`` frames, so successive chunks share
    ``overlap`` frames at their boundaries.

    Parameters
    ----------
    image_dir : str or Path
        Directory containing PNG images, sorted lexicographically.
    T_batch_size : int
        Number of frames per chunk (temporal batch size).
    start_frame : int
        Skip this many leading frames.
    total_frames : int or None
        Process at most this many frames after ``start_frame``
        (``None`` = all available).
    overlap : int
        Frames shared between consecutive chunks.
    image_end : str
        Filename suffix used to glob images (default ``".png"``).
    fixed_frame : int
        If >= 0, every slot in every chunk is filled with this one frame
        (useful for single-image relighting experiments).
    resolution : (H, W)
        Target resolution for loaded images.
    apply_blur : bool
        If True, apply a small Gaussian blur to the basecolor channel.
    """

    def __init__(
        self,
        image_dir: str,
        T_batch_size: int = 57,
        start_frame: int = 0,
        total_frames: Optional[int] = None,
        overlap: int = 32,
        image_end: str = ".png",
        fixed_frame: int = -1,
        resolution: Tuple[int, int] = (704, 1280),
        apply_blur: bool = False,
    ):
        self.path        = Path(image_dir)
        self.T           = T_batch_size
        self.overlap     = overlap
        self.fixed_frame = fixed_frame
        self.resolution  = resolution
        self.apply_blur  = apply_blur

        all_paths = sorted(self.path.glob(f"*{image_end}"))
        all_paths = all_paths[start_frame:]
        if total_frames is not None:
            assert len(all_paths) >= total_frames, (
                f"Requested {total_frames} frames but only {len(all_paths)} available "
                f"in {image_dir} after start_frame={start_frame}."
            )
            all_paths = all_paths[:total_frames]

        self.image_paths = all_paths
        self.total_frames = len(all_paths)
        self.chunk_pointer = 0

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of full chunks available."""
        if self.total_frames < self.T:
            return 0
        step = self.T - self.overlap
        return 1 + (self.total_frames - self.T) // step

    def reset(self) -> None:
        """Rewind to the first frame."""
        self.chunk_pointer = 0

    # ------------------------------------------------------------------

    def _load_single_frame(self, img_path: Path, frame_id: int) -> Dict:
        """Load one image file into a single-sample data-batch dict (no B dim)."""
        H, W = self.resolution
        loaded = load_modalities(
            {"video": str(img_path)},
            pad_with_zero=True,
            H=H, W=W,
            num_frames=1,
        )
        # Clamp near-black values (dataset convention)
        loaded["video"][0][loaded["video"][0] < -0.99] = -1.0

        data = {
            "frame_id": torch.tensor([frame_id], dtype=torch.long),
            "video":    loaded["video"][0],  # (C, T=1, H, W)
        }
        data["basecolor"] = data["video"].clone()

        if self.apply_blur:
            import torchvision
            data["basecolor"] = torchvision.transforms.functional.gaussian_blur(
                data["basecolor"], kernel_size=(3, 3), sigma=(0.5, 0.5)
            )

        single = prepare_base_data_batch(data, num_frames=1, resolution=self.resolution, with_dummy_batch=False)
        # Add batch dimension so concatenate_data_batches receives (B=1, ...) tensors
        return {k: v.unsqueeze(0) for k, v in single.items()}

    # ------------------------------------------------------------------

    def next_chunk(self) -> Tuple[bool, Optional[Dict]]:
        """Return the next temporal chunk.

        Returns
        -------
        found : bool
            False when fewer than ``T_batch_size`` frames remain.
        result : dict or None
            ``frame_ids``    – list of frame indices in this chunk
            ``data_batch``   – complete data-batch dict, video shape (1, C, T, H, W)
            ``frame_name_0`` – stem of the first frame's filename
        """
        if self.chunk_pointer + self.T > self.total_frames:
            return False, None

        frame_ids = list(range(self.chunk_pointer, self.chunk_pointer + self.T))
        paths = self.image_paths[self.chunk_pointer : self.chunk_pointer + self.T]
        if self.fixed_frame >= 0:
            paths = [self.image_paths[self.fixed_frame]] * self.T

        batches = [
            self._load_single_frame(p, fid)
            for p, fid in zip(paths, frame_ids)
        ]
        data_batch = concatenate_data_batches(batches, num_frames=self.T)

        step = self.T - self.overlap
        print(
            f"Advancing chunk pointer {self.chunk_pointer} → {self.chunk_pointer + step} …",
            flush=True,
        )
        self.chunk_pointer += step

        return True, {
            "frame_ids":    frame_ids,
            "data_batch":   data_batch,
            "frame_name_0": paths[0].stem.split(".")[0],
        }


# ---------------------------------------------------------------------------
# Single-env-map loader
# ---------------------------------------------------------------------------

class BatchedEnvLoaderSingle:
    """Project a single HDR environment map to camera space for each frame.

    Camera poses can come from three sources (in priority order):
    1. A JSON trajectory file (``trajectory_file``).
    2. A calibration directory (``calib_dir``) + ``cam_id`` at load time.
    3. No pose (env map processed without camera rotation).

    Parameters
    ----------
    envmap_path : str or Path
        Path to the HDR environment map.
    trajectory_file : str or None
        JSON file with a list of ``{"extrinsics": [[…]]}`` entries.
        Row permutation ``[[1,2,0,3], :]`` plus OpenGL→OpenCV conversion
        is applied to each matrix (matches the EVA trajectory convention).
    calib_dir : str or None
        Directory containing a ``cameras.calib`` file.
    device : str
    strength : float
        Environment map intensity multiplier.
    fixed_frames : int
        If >= 0, all frame indices < ``fixed_frames`` use pose index 0;
        indices >= ``fixed_frames`` use ``index - fixed_frames``.
    resolution : (H, W)
    """

    def __init__(
        self,
        envmap_path: str,
        trajectory_file: Optional[str] = None,
        calib_dir: Optional[str] = None,
        device: str = "cuda",
        strength: float = 1.0,
        fixed_frames: int = -1,
        resolution: Tuple[int, int] = (704, 1280),
    ):
        self.envmap_path  = str(envmap_path)
        self.device       = device
        self.env_strength = strength
        self.fixed_frames = fixed_frames
        self.resolution   = resolution

        self.poses      = None
        self.extrinsics = None

        if calib_dir is not None:
            intrinsics: List = []
            extrinsics: List = []
            read_calib(str(calib_dir), IMAGE_W=1000, INTR=intrinsics, EXTR=extrinsics, invert_extr=True)
            self.extrinsics = extrinsics

        if trajectory_file is not None:
            self.poses = self._load_poses(trajectory_file)

    # ------------------------------------------------------------------

    @staticmethod
    def _load_poses(trajectory_file: str) -> List:
        """Load camera poses from a JSON trajectory file.

        Each entry must have an ``"extrinsics"`` key containing a 4×4 matrix.
        The row permutation ``[[1,2,0,3], :]`` (EVA convention) followed by
        OpenGL→OpenCV conversion is applied before storing.
        """
        with open(trajectory_file) as f:
            data = json.load(f)

        poses = []
        for entry in data:
            ext = np.array(entry["extrinsics"], dtype=np.float64)
            ext = ext[[1, 2, 0, 3], :]  # undo EVA left-shift
            ext = convert_c2w_opengl_to_opencv(ext)
            poses.append(ext)
        return poses

    # ------------------------------------------------------------------

    def load_envmaps(self, frame_ids: List[int], cam_id: Optional[int] = None) -> Dict:
        """Build env-map tensors for a list of frame indices.

        Parameters
        ----------
        frame_ids : list of int
            Frame indices to process (length = T).
        cam_id : int or None
            Camera index; used when falling back to calibration-based poses.

        Returns
        -------
        dict with keys ``env_ldr``, ``env_log``, ``env_nrm`` — each (1, C, T, H, W).
        """
        T = len(frame_ids)

        # Resolve per-frame pose indices
        if self.fixed_frames >= 0:
            pose_ids = [0 if fid < self.fixed_frames else fid - self.fixed_frames
                        for fid in frame_ids]
        else:
            pose_ids = list(frame_ids)

        # Build pose list
        if self.poses is not None:
            pose_list = [self.poses[pid % len(self.poses)] for pid in pose_ids]
        elif self.extrinsics is not None and cam_id is not None:
            pose_list = [convert_c2w_opengl_to_opencv(self.extrinsics[cam_id])] * T
        else:
            pose_list = None

        envlight = process_environment_map(
            self.envmap_path,
            resolution=self.resolution,
            num_frames=T,
            fixed_pose=True,
            pose_list=pose_list,
            rotate_envlight=False,
            env_format=["proj"],
            device=self.device,
            env_strength=self.env_strength,
            env_flip=False,
        )

        env_ldr = envlight["env_ldr"].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1  # (1,C,T,H,W)
        env_log = envlight["env_log"].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
        env_nrm = envmap_vec(list(self.resolution), device=self.device)             # (H,W,3)
        env_nrm = env_nrm.unsqueeze(0).unsqueeze(0).permute(0, 4, 1, 2, 3).expand_as(env_ldr)

        return {"env_ldr": env_ldr, "env_log": env_log, "env_nrm": env_nrm}


# ---------------------------------------------------------------------------
# Dataset-based env-map loader
# ---------------------------------------------------------------------------

class BatchedEnvLoaderDataset:
    """Load per-frame env maps from an ERA-style dataset directory.

    Expected layout::

        {dataset_dir}/Frame_{frame_id:06d}/{envmap_name}
        {dataset_dir}/shared/cameras.calib     (or camera.calib)

    Parameters
    ----------
    dataset_dir : str or Path
        Root of the ERA dataset.
    envmap_name : str
        Filename of the env map inside each frame directory.
    resolution : (H, W)
    device : str
    """

    def __init__(
        self,
        dataset_dir: str,
        envmap_name: str = "EnvMapSampled.hdr",
        resolution: Tuple[int, int] = (704, 1280),
        device: str = "cuda",
    ):
        self.dataset_dir = Path(dataset_dir)
        self.envmap_name = envmap_name
        self.resolution  = resolution
        self.device      = device

        intrinsics: List = []
        extrinsics: List = []
        read_calib(
            str(self.dataset_dir / "shared"),
            IMAGE_W=1000,
            INTR=intrinsics,
            EXTR=extrinsics,
            invert_extr=True,
        )
        self.extrinsics = extrinsics

    # ------------------------------------------------------------------

    def load_envmaps(self, frame_id: int, cam_id: int) -> Dict:
        """Load the env map for ``frame_id`` projected to ``cam_id``'s pose.

        Returns
        -------
        dict with keys ``env_ldr``, ``env_log``, ``env_nrm`` — each (1, C, 1, H, W).
        """
        envmap_path = self.dataset_dir / f"Frame_{frame_id:06d}" / self.envmap_name
        pose_list   = [convert_c2w_opengl_to_opencv(self.extrinsics[cam_id])]

        envlight = process_environment_map(
            str(envmap_path),
            resolution=self.resolution,
            num_frames=1,
            fixed_pose=True,
            pose_list=pose_list,
            rotate_envlight=False,
            env_format=["proj"],
            device=self.device,
            env_strength=1.0,
            env_flip=False,
        )

        env_ldr = envlight["env_ldr"].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1  # (1,C,1,H,W)
        env_log = envlight["env_log"].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
        env_nrm = envmap_vec(list(self.resolution), device=self.device)
        env_nrm = env_nrm.unsqueeze(0).unsqueeze(0).permute(0, 4, 1, 2, 3).expand_as(env_ldr)

        return {"env_ldr": env_ldr, "env_log": env_log, "env_nrm": env_nrm}
