"""Tiled Real-ESRGAN x4plus upscaler via spandrel (BSD-3) for 16GB Apple Silicon Mac."""

import logging
import os
from typing import Tuple, Optional
from PIL import Image, ImageFilter
import numpy as np
import torch

logger = logging.getLogger(__name__)

_UPSCALER_MODEL = None
_DEFAULT_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "RealESRGAN_x4plus.pth")


def get_upscale_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_upscaler(weights_path: str = _DEFAULT_WEIGHTS_PATH):
    """Load Real-ESRGAN x4plus via spandrel."""
    global _UPSCALER_MODEL
    if _UPSCALER_MODEL is not None:
        return _UPSCALER_MODEL

    if not os.path.exists(weights_path):
        # Check standard cache paths
        alt_paths = [
            os.path.expanduser("~/.cache/saadaa/RealESRGAN_x4plus.pth"),
            "/tmp/RealESRGAN_x4plus.pth",
            "RealESRGAN_x4plus.pth"
        ]
        for p in alt_paths:
            if os.path.exists(p):
                weights_path = p
                break

    if not os.path.exists(weights_path):
        logger.info(f"Real-ESRGAN weights not found at {weights_path}. Using Lanczos + unsharp mask fallback.")
        return None

    try:
        import spandrel
        loader = spandrel.ModelLoader()
        descriptor = loader.load_from_file(weights_path)
        device = get_upscale_device()
        descriptor = descriptor.to(device).eval()
        _UPSCALER_MODEL = descriptor
        logger.info(f"Loaded Real-ESRGAN x4plus on {device} via spandrel.")
        return _UPSCALER_MODEL
    except Exception as e:
        logger.error(f"Failed to load Real-ESRGAN with spandrel: {e}")
        return None


def tile_upscale(
    image: Image.Image,
    descriptor,
    tile_size: int = 512,
    tile_overlap: int = 32
) -> Image.Image:
    """Perform tiled inference with Real-ESRGAN to keep peak VRAM under 2GB on 16GB Mac."""
    device = get_upscale_device()
    scale = descriptor.scale
    
    img_np = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    h, w, c = img_np.shape
    
    out_h = h * scale
    out_w = w * scale
    
    output_np = np.zeros((out_h, out_w, c), dtype=np.float32)
    weight_np = np.zeros((out_h, out_w, c), dtype=np.float32)

    y_stride = tile_size - tile_overlap
    x_stride = tile_size - tile_overlap

    y_points = list(range(0, h, y_stride))
    x_points = list(range(0, w, x_stride))

    for y in y_points:
        y1 = y
        y2 = min(h, y + tile_size)
        y1 = max(0, y2 - tile_size)
        
        for x in x_points:
            x1 = x
            x2 = min(w, x + tile_size)
            x1 = max(0, x2 - tile_size)

            tile_in = img_np[y1:y2, x1:x2]
            # Convert to [1, C, H, W]
            tensor_in = torch.from_numpy(tile_in).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                tensor_out = descriptor(tensor_in)
                tile_out = tensor_out.squeeze(0).permute(1, 2, 0).cpu().numpy()

            out_y1 = y1 * scale
            out_y2 = y2 * scale
            out_x1 = x1 * scale
            out_x2 = x2 * scale

            # Linear weight ramp for seamless stitching
            th, tw = tile_out.shape[:2]
            mask_y = np.ones((th, 1, 1), dtype=np.float32)
            mask_x = np.ones((1, tw, 1), dtype=np.float32)
            
            ramp_len = tile_overlap * scale
            if y1 > 0 and ramp_len > 0:
                mask_y[:ramp_len, 0, 0] = np.linspace(0.01, 1.0, ramp_len)
            if y2 < h and ramp_len > 0:
                mask_y[-ramp_len:, 0, 0] = np.linspace(1.0, 0.01, ramp_len)
            if x1 > 0 and ramp_len > 0:
                mask_x[0, :ramp_len, 0] = np.linspace(0.01, 1.0, ramp_len)
            if x2 < w and ramp_len > 0:
                mask_x[0, -ramp_len:, 0] = np.linspace(1.0, 0.01, ramp_len)
                
            tile_mask = mask_y * mask_x
            output_np[out_y1:out_y2, out_x1:out_x2] += tile_out * tile_mask
            weight_np[out_y1:out_y2, out_x1:out_x2] += tile_mask

    weight_np = np.maximum(weight_np, 1e-4)
    final_np = np.clip((output_np / weight_np) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(final_np)


def upscale_to_channel_size(
    image: Image.Image,
    target_size: Tuple[int, int],
    quality_preset: str = "standard"
) -> Image.Image:
    """Upscale image to target channel dimensions.
    
    - 'draft': Fast Lanczos resize without ESRGAN (for speed).
    - 'standard' / 'best': Real-ESRGAN x4plus (tiled) + Lanczos downscale to exact channel dimension.
    """
    target_w, target_h = target_size
    current_w, current_h = image.size

    # If already at or above target size, high quality Lanczos resize
    if current_w >= target_w and current_h >= target_h:
        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    if quality_preset == "draft":
        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # Try Real-ESRGAN
    descriptor = load_upscaler()
    if descriptor is not None:
        try:
            upscaled = tile_upscale(image, descriptor, tile_size=512, tile_overlap=32)
            # Resize from 4x down to exact target dimensions
            return upscaled.resize((target_w, target_h), Image.Resampling.LANCZOS)
        except Exception as e:
            logger.error(f"Real-ESRGAN inference failed: {e}. Falling back to Lanczos.")

    # High quality fallback: Lanczos + subtle unsharp mask sharpening
    resized = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    sharpened = resized.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))
    return sharpened
