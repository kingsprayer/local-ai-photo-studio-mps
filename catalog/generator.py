"""SDXL inpainting, ControlNet OpenPose, and SDXL-Lightning generator optimized for 16GB Apple Silicon Mac."""

import logging
import os
import gc
import math
from typing import Dict, Any, Optional, Tuple, Union
import torch
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Model identifiers (all commercial-use licensed)
SDXL_INPAINT_ID = "OzzyGT/RealVisXL_V4.0_inpainting"
VAE_FIX_ID = "madebyollin/sdxl-vae-fp16-fix"
CONTROLNET_OPENPOSE_ID = "xinsir/controlnet-openpose-sdxl-1.0"
LIGHTNING_LORA_ID = "ByteDance/SDXL-Lightning"
LIGHTNING_LORA_FILE = "sdxl_lightning_8step_lora.safetensors"

# Singleton pipelines
_BASE_PIPELINE = None
_CONTROLNET_MODEL = None
_CONTROLNET_PIPELINE = None
_CURRENT_MODE = None  # 'standard', 'draft'


def get_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_memory():
    """Clear memory caches on Apple Silicon Mac."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_sdxl_base_pipeline(is_draft: bool = False):
    """Load SDXL inpainting pipeline with madebyollin fp16 fixed VAE."""
    global _BASE_PIPELINE, _CURRENT_MODE
    from diffusers import StableDiffusionXLInpaintPipeline, AutoencoderKL, EulerDiscreteScheduler

    device = get_torch_device()
    dtype = torch.float16 if device.type == "mps" else torch.float32

    if _BASE_PIPELINE is None:
        logger.info(f"Loading SDXL base inpainting pipeline ({SDXL_INPAINT_ID}) on {device}...")
        try:
            vae = AutoencoderKL.from_pretrained(VAE_FIX_ID, torch_dtype=dtype)
            _BASE_PIPELINE = StableDiffusionXLInpaintPipeline.from_pretrained(
                SDXL_INPAINT_ID,
                vae=vae,
                torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None,
                use_safetensors=True
            ).to(device)
            _BASE_PIPELINE.enable_attention_slicing()
        except Exception as e:
            logger.error(f"Failed to load SDXL pipeline from hub: {e}. Check internet/weights.")
            raise

    # Handle Draft mode vs Standard mode
    target_mode = "draft" if is_draft else "standard"
    if _CURRENT_MODE != target_mode:
        if is_draft:
            logger.info("Configuring SDXL-Lightning 8-step LoRA with trailing Euler scheduler...")
            try:
                _BASE_PIPELINE.scheduler = EulerDiscreteScheduler.from_config(
                    _BASE_PIPELINE.scheduler.config,
                    timestep_spacing="trailing"
                )
                _BASE_PIPELINE.load_lora_weights(LIGHTNING_LORA_ID, weight_name=LIGHTNING_LORA_FILE)
                _CURRENT_MODE = "draft"
            except Exception as e:
                logger.warning(f"Failed to load SDXL-Lightning LoRA ({e}), using standard scheduler.")
                _CURRENT_MODE = "standard"
        else:
            if _CURRENT_MODE == "draft":
                logger.info("Unloading LoRA weights for standard high-fidelity generation...")
                try:
                    _BASE_PIPELINE.unload_lora_weights()
                except Exception:
                    pass
            _BASE_PIPELINE.scheduler = EulerDiscreteScheduler.from_config(_BASE_PIPELINE.scheduler.config)
            _CURRENT_MODE = "standard"

    return _BASE_PIPELINE


def load_controlnet_pipeline(is_draft: bool = False):
    """Build ControlNet inpainting pipeline from base pipeline components to share weights in memory."""
    global _CONTROLNET_PIPELINE, _CONTROLNET_MODEL
    from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel

    device = get_torch_device()
    dtype = torch.float16 if device.type == "mps" else torch.float32

    base_pipe = load_sdxl_base_pipeline(is_draft=is_draft)

    if _CONTROLNET_MODEL is None:
        logger.info(f"Loading ControlNet OpenPose model ({CONTROLNET_OPENPOSE_ID})...")
        _CONTROLNET_MODEL = ControlNetModel.from_pretrained(
            CONTROLNET_OPENPOSE_ID,
            torch_dtype=dtype,
            use_safetensors=True
        ).to(device)

    if _CONTROLNET_PIPELINE is None:
        logger.info("Assembling StableDiffusionXLControlNetInpaintPipeline sharing components with base pipeline...")
        _CONTROLNET_PIPELINE = StableDiffusionXLControlNetInpaintPipeline(
            vae=base_pipe.vae,
            text_encoder=base_pipe.text_encoder,
            text_encoder_2=base_pipe.text_encoder_2,
            tokenizer=base_pipe.tokenizer,
            tokenizer_2=base_pipe.tokenizer_2,
            unet=base_pipe.unet,
            controlnet=_CONTROLNET_MODEL,
            scheduler=base_pipe.scheduler
        ).to(device)
        _CONTROLNET_PIPELINE.enable_attention_slicing()

    return _CONTROLNET_PIPELINE


def run_generation(
    init_image: Image.Image,
    mask_image: Image.Image,
    prompt: str,
    prompt_2: str,
    negative_prompt: str,
    pose_image: Optional[Image.Image] = None,
    quality_preset: str = "standard",
    strength: float = 1.0,
    seed: Optional[int] = None,
    progress_callback=None,
    mock_mode: bool = False
) -> Image.Image:
    """Execute SDXL inpainting generation with memory-safe MPS optimization.
    
    Guarantees:
    - Start diffusion at strength 1.0 (not 0.9999) or placeholder colour leaks.
    - Guarantee >= 4 effective denoising steps when strength < 1.
    """
    is_draft = (quality_preset == "draft")
    
    # 1. Determine steps and CFG
    if is_draft:
        steps = 8
        cfg = 1.0  # CFG off for SDXL-Lightning
    elif quality_preset == "best":
        steps = 30
        cfg = 6.5
    else:  # standard
        steps = 25
        cfg = 6.0

    # 2. Strict strength guarantee: if strength < 1.0, ensure >= 4 effective denoising steps
    effective_steps = int(steps * strength)
    if strength < 1.0 and effective_steps < 4:
        # Scale steps up so effective steps >= 4
        steps = int(math.ceil(4.0 / max(0.01, strength)))
        logger.info(f"Scaled inference steps to {steps} to guarantee >= 4 effective denoising steps.")

    # 3. Seed management
    generator = None
    if seed is not None:
        device = get_torch_device()
        generator = torch.Generator(device=device).manual_seed(seed)

    # 4. Mock / Smoke test mode: runs synthetic diffusion without needing multi-gigabyte weights
    if mock_mode or os.environ.get("SAADAA_MOCK_DIFFUSION") == "1":
        logger.info("Executing in mock diffusion mode for test/validation...")
        w, h = init_image.size
        # Generate a realistic mock output:
        # Blend init image with neutral studio lighting and subtle textures
        init_np = np.array(init_image.convert("RGB"), dtype=np.float32)
        mask_np = np.array(mask_image.convert("L"), dtype=np.float32) / 255.0
        mask_3d = np.expand_dims(mask_np, axis=2)

        # Create smooth background tint (warm ivory/limestone)
        synthetic_bg = np.full((h, w, 3), [246, 244, 238], dtype=np.float32)
        # Add subtle vertical studio gradient
        for y in range(h):
            factor = 1.0 - (y / float(h)) * 0.08
            synthetic_bg[y, :, :] *= factor

        # Blend where mask is inpaint
        blended = init_np * (1.0 - mask_3d) + synthetic_bg * mask_3d
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    # 5. Full Real Model Inference
    try:
        if pose_image is not None:
            # ControlNet pipeline
            pipe = load_controlnet_pipeline(is_draft=is_draft)
            output = pipe(
                prompt=prompt,
                prompt_2=prompt_2,
                negative_prompt=negative_prompt,
                image=init_image,
                mask_image=mask_image,
                control_image=pose_image,
                num_inference_steps=steps,
                guidance_scale=cfg,
                strength=strength,
                generator=generator
            )
        else:
            # Standard inpainting pipeline
            pipe = load_sdxl_base_pipeline(is_draft=is_draft)
            output = pipe(
                prompt=prompt,
                prompt_2=prompt_2,
                negative_prompt=negative_prompt,
                image=init_image,
                mask_image=mask_image,
                num_inference_steps=steps,
                guidance_scale=cfg,
                strength=strength,
                generator=generator
            )

        clear_memory()
        return output.images[0]
    except Exception as e:
        logger.error(f"Diffusion generation failed: {e}")
        clear_memory()
        raise
