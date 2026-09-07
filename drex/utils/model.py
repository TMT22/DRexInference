"""Model loading, LoRA setup, and checkpoint utilities."""

import torch
from typing import Optional

from cosmos_predict1.diffusion.inference.inference_utils import (
    load_model_by_config,
    load_network_model,
    load_tokenizer_model,
)
from cosmos_predict1.diffusion.model.model_diffusion_renderer import DiffusionRendererModel
from cosmos_predict1.diffusion.training.utils.layer_control.peft_control_config_parser import LayerControlConfigParser
from cosmos_predict1.diffusion.training.utils.peft.peft import add_lora_layers, setup_lora_requires_grad

from cosmos_predict1.utils import log


DR_INVERSE_NAME = "Diffusion_Renderer_Inverse_Cosmos_7B"
DR_FORWARD_NAME = "Diffusion_Renderer_Forward_Cosmos_7B"
TOKENIZER_NAME  = "Cosmos-Tokenize1-CV8x8x8-720p"

_DR_INVERSE_MODEL = None
_DR_FORWARD_MODEL = None


def get_model(name: str, checkpoint_dir: str = "cosmos_transfer1/checkpoints") -> DiffusionRendererModel:
    """Lazily load and cache a pre-trained DiffusionRenderer model.

    Parameters
    ----------
    name : str
        One of DR_INVERSE_NAME or DR_FORWARD_NAME.
    checkpoint_dir : str
        Directory containing the model sub-directories and tokenizer.
    """
    global _DR_INVERSE_MODEL, _DR_FORWARD_MODEL

    if name == DR_INVERSE_NAME:
        if _DR_INVERSE_MODEL is None:
            log.info("Loading inverse renderer model...")
            _DR_INVERSE_MODEL = load_model_by_config(
                config_job_name=DR_INVERSE_NAME,
                config_file="cosmos_transfer1/cosmos_predict1/diffusion/config/diffusion_renderer_config.py",
                model_class=DiffusionRendererModel,
            )
            load_network_model(_DR_INVERSE_MODEL, f"{checkpoint_dir}/{DR_INVERSE_NAME}/model.pt")
            load_tokenizer_model(_DR_INVERSE_MODEL, f"{checkpoint_dir}/{TOKENIZER_NAME}")
        return _DR_INVERSE_MODEL

    if name == DR_FORWARD_NAME:
        if _DR_FORWARD_MODEL is None:
            log.info("Loading forward renderer model...")
            _DR_FORWARD_MODEL = load_model_by_config(
                config_job_name=DR_FORWARD_NAME,
                config_file="cosmos_transfer1/cosmos_predict1/diffusion/config/diffusion_renderer_config.py",
                model_class=DiffusionRendererModel,
            )
            load_network_model(_DR_FORWARD_MODEL, f"{checkpoint_dir}/{DR_FORWARD_NAME}/model.pt")
            load_tokenizer_model(_DR_FORWARD_MODEL, f"{checkpoint_dir}/{TOKENIZER_NAME}")
        return _DR_FORWARD_MODEL

    raise ValueError(f"Unknown model name: {name!r}")


def model_setup_lora(model: DiffusionRendererModel, lora_config: dict) -> DiffusionRendererModel:
    """Add LoRA layers to model in-place and return the model."""
    parser = LayerControlConfigParser(config=lora_config)
    add_lora_layers(model, parser.parse())
    return model


def load_lora_checkpoint(
    model: DiffusionRendererModel,
    checkpoint_path: str,
    strict_resume: bool = False,
) -> DiffusionRendererModel:
    """Load LoRA parameters from a checkpoint file into model."""
    tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}
    state_dict = torch.load(checkpoint_path, map_location="cuda")
    lora_state = {k: v.to(**tensor_kwargs) for k, v in state_dict["model"].items()}
    info = model.load_state_dict(lora_state, strict=strict_resume)
    log.info(f"Loaded LoRA checkpoint from {checkpoint_path}: {info}")
    return model


def load_model_for_inference(
    checkpoint_type: str,
    checkpoint_dir: str = "cosmos_transfer1/checkpoints",
    checkpoint_path: Optional[str] = None,
    lora_config: Optional[dict] = None,
    strict_resume: bool = False,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> DiffusionRendererModel:
    """Load the forward DR model ready for inference.

    Parameters
    ----------
    checkpoint_type : "lora" | "zero-shot"
    checkpoint_dir : str
        Directory containing pre-trained base weights (model.pt + tokenizer).
    checkpoint_path : str or None
        Path to LoRA checkpoint file.  Required for checkpoint_type="lora".
    lora_config : dict or None
        LoRA layer config (from ``TrainerConfig.build_lora_config()``).
        Required for checkpoint_type="lora".
    strict_resume : bool
        Passed to the checkpoint loader.
    device, dtype : target device / dtype for the loaded model.
    """
    model = get_model(DR_FORWARD_NAME, checkpoint_dir=checkpoint_dir)
    model = model.to(device=device, dtype=dtype)

    if checkpoint_type == "zero-shot":
        return model

    if checkpoint_type == "lora":
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for checkpoint_type='lora'")
        if lora_config is None:
            raise ValueError("lora_config is required for checkpoint_type='lora'")
        model = model_setup_lora(model, lora_config)
        model = load_lora_checkpoint(model, checkpoint_path, strict_resume)
    else:
        raise ValueError(f"Unknown checkpoint_type {checkpoint_type!r}. Expected 'lora' or 'zero-shot'.")

    # LoRA layers are added as new nn.Modules after the initial .to() call, so
    # they initialize on CPU.  A second .to() guarantees every parameter ends up
    # on the correct device/dtype regardless of checkpoint coverage.
    model = model.to(device=device, dtype=dtype)
    return model
