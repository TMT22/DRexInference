"""Batch assembly and diffusion-renderer inference utilities."""

import torch
import numpy as np
from typing import Dict, List

from cosmos_predict1.diffusion.model.model_diffusion_renderer import DiffusionRendererModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GBUFFER_INDEX_MAPPING: Dict[str, int] = {
    "basecolor":      0,
    "metallic":       1,
    "roughness":      2,
    "normal":         3,
    "depth":          4,
    "diffuse_albedo": 5,
    "specular_albedo":6,
}


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------

def prepare_base_data_batch(
    additional_entries: Dict[str, torch.Tensor],
    resolution: tuple = (704, 1280),
    num_frames: int = 1,
    fps: float = 24.0,
    with_dummy_batch: bool = True,
) -> Dict[str, torch.Tensor]:
    """Build a minimal data-batch dict accepted by DiffusionRendererModel."""
    dummy_t5_emb  = torch.zeros(512, 1024)
    dummy_t5_mask = torch.zeros(512)
    dummy_t5_mask[0] = 1.0

    if with_dummy_batch:
        batch = {
            "context_index":      torch.LongTensor([0]).unsqueeze(0),
            "t5_text_embeddings": dummy_t5_emb.unsqueeze(0),
            "t5_text_mask":       dummy_t5_mask.unsqueeze(0),
            "fps":                torch.tensor([fps], dtype=torch.float),
            "image_size":         torch.from_numpy(np.asarray(resolution)).unsqueeze(0),
            "padding_mask":       torch.zeros(1, resolution[0], resolution[1]).unsqueeze(0),
            "num_frames":         torch.tensor([num_frames], dtype=torch.float),
        }
    else:
        batch = {
            "context_index":      torch.LongTensor([0]),
            "t5_text_embeddings": dummy_t5_emb,
            "t5_text_mask":       dummy_t5_mask,
            "fps":                torch.tensor([fps], dtype=torch.float),
            "image_size":         torch.from_numpy(np.asarray(resolution)),
            "padding_mask":       torch.zeros(1, resolution[0], resolution[1]),
            "num_frames":         torch.tensor([num_frames], dtype=torch.float),
        }

    batch.update(additional_entries)
    return batch


def concatenate_data_batches(
    batch_list: List[Dict],
    num_frames: int = 57,
    envmap_handle: str = "extend_start",
) -> Dict[str, torch.Tensor]:
    """Concatenate single-frame batches along the time dimension."""
    env_keys = {"env_nrm", "env_ldr", "env_log"}

    while len(batch_list) < num_frames:
        batch_list.append(batch_list[-1])

    out: Dict[str, torch.Tensor] = {}
    for key in batch_list[0]:
        if key in env_keys:
            if envmap_handle == "ignore":
                continue
            if envmap_handle == "extend_start":
                out[key] = batch_list[0][key].repeat(1, 1, num_frames, 1, 1)
                continue
        if not isinstance(batch_list[0][key], torch.Tensor) or batch_list[0][key].dim() != 5:
            out[key] = batch_list[0][key]
        else:
            out[key] = torch.cat([b[key] for b in batch_list], dim=2)

    device = out[next(k for k in out if isinstance(out[k], torch.Tensor))].device
    out["num_frames"] = torch.tensor([num_frames], dtype=torch.float, device=device).unsqueeze(0)
    return out


# ---------------------------------------------------------------------------
# Inference utilities
# ---------------------------------------------------------------------------

def move_batch_to_device(batch: Dict, device) -> Dict:
    """Return a shallow copy of *batch* with every tensor moved to *device*."""
    return {
        k: (v.to(device=device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def _decode_with_normal_blend(model, sample, normalize_normal: bool) -> torch.Tensor:
    video = model.decode(sample)
    if normalize_normal:
        norm = torch.norm(video, dim=1, p=2, keepdim=True)
        video_norm = video / norm.clamp(min=1e-12)
        lo, hi = 0.2, 0.4
        blend = torch.clip((norm - lo) / (hi - lo), 0.0, 1.0)
        video = video_norm * blend + video * (1.0 - blend)
    return video


def run_custom_diffrender_model(
    dr_model: DiffusionRendererModel,
    data_batch: Dict,
    normalize_normal: bool = False,
    seed: int = 42,
    guidance: float = 0.0,
    num_steps: int = 15,
    is_negative_prompt: bool = False,
) -> torch.Tensor:
    """Run an already-loaded DiffusionRenderer model.

    Returns (B, 3, T, H, W) in [-1, 1].
    """
    device = next(dr_model.parameters()).device
    data_batch = move_batch_to_device(data_batch, device)

    C = dr_model.tokenizer.channel
    T = data_batch["video"].shape[2]
    H = data_batch["video"].shape[3] // dr_model.tokenizer.spatial_compression_factor
    W = data_batch["video"].shape[4] // dr_model.tokenizer.spatial_compression_factor
    state_shape = [C, (T - 1) // 8 + 1, H, W]

    with torch.no_grad():
        sample = dr_model.generate_samples_from_batch(
            data_batch,
            guidance=guidance,
            state_shape=state_shape,
            num_steps=num_steps,
            is_negative_prompt=is_negative_prompt,
            seed=seed,
        )
    return _decode_with_normal_blend(dr_model, sample, normalize_normal)
