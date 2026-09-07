"""Minimal config for D-Rex inference: LoRA settings and checkpoint paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type, TypeVar

import yaml


T = TypeVar("T")


def load_config(path: str, cls: Type[T], overrides: dict = None) -> T:
    """Load a YAML file and populate a dataclass of type *cls*.

    Unknown YAML keys are silently ignored; missing keys keep dataclass defaults.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    if overrides:
        raw.update(overrides)

    known_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in raw.items() if k in known_fields}
    return cls(**filtered)


@dataclass
class TrainerConfig:
    """Inference-relevant subset of training config (LoRA + checkpoint location)."""

    checkpoint_dir:     str   = "cosmos_transfer1/checkpoints"
    lora_rank:          int   = 8
    lora_scale:         float = 1.0
    lora_first_nblocks: int   = 28

    def build_lora_config(self) -> dict:
        from cosmos_predict1.diffusion.training.utils.peft.lora_config import get_fa_ca_qv_lora_config
        return get_fa_ca_qv_lora_config(
            first_nblocks=self.lora_first_nblocks,
            rank=self.lora_rank,
            scale=self.lora_scale,
        )
