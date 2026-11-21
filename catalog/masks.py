"""Mask generation, BiRefNet person matting, erosion/trimming, and inpaint mask construction."""

import logging
from typing import Tuple, Optional, Union
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2
import rembg

logger = logging.getLogger(__name__)

# Cached rembg session
_REMBG_SESSION = None

def get_rembg_session():
    """Lazily initialize rembg session with birefnet-general-lite or fallback."""
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        try:
            logger.info("Initializing rembg session with birefnet-general-lite...")
            _REMBG_SESSION = rembg.new_session("birefnet-general-lite")
        except Exception as e:
            logger.warning(f"Could not load birefnet-general-lite ({e}), falling back to default rembg session...")
            try:
                _REMBG_SESSION = rembg.new_session("u2net")
            except Exception as e2:
                logger.error(f"Fallback to u2net also failed: {e2}")
                _REMBG_SESSION = False
    return _REMBG_SESSION


def extract_subject_matte(image: Image.Image) -> Image.Image:
    """Extract a clean grayscale matte (0-255) of the subject (person or product) using BiRefNet."""
    session = get_rembg_session()
    if session and session is not False:
        try:
            # rembg returns RGBA
            output_rgba = rembg.remove(image, session=session)
            # Alpha channel is the precise matte
            matte = output_rgba.split()[3]
            return matte
        except Exception as e:
            logger.error(f"rembg extraction failed: {e}")
    
    # Graceful fallback: threshold on luminance/contrast from center
    logger.info("Using heuristic contrast-based matte fallback...")
    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    # Morphological clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return Image.fromarray(cleaned)


def trim_mask(mask: Image.Image, px: int = 2) -> Image.Image:
    """Trim pixels off garment/subject mask edges using erosion.
    
    Crucial requirement: Trimming 2px completely removes the old-background fringe
    and color bleed when compositing the kept garment back over the new image.
    """
    if px <= 0:
        return mask
    mask_np = np.array(mask.convert("L"))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    eroded = cv2.erode(mask_np, kernel, iterations=1)
    return Image.fromarray(eroded)


def expand_mask(mask: Image.Image, px: int = 4) -> Image.Image:
    """Expand inpaint mask edges using dilation to ensure seamless blending."""
    if px <= 0:
        return mask
    mask_np = np.array(mask.convert("L"))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    dilated = cv2.dilate(mask_np, kernel, iterations=1)
    return Image.fromarray(dilated)


def feather_mask(mask: Image.Image, radius: float = 2.0) -> Image.Image:
    """Softly feather mask edges with Gaussian blur for smooth alpha blending."""
    return mask.filter(ImageFilter.GaussianBlur(radius))


def create_inpaint_mask(
    mode: str,
    subject_matte: Image.Image,
    garment_mask: Optional[Image.Image] = None,
    keep_mode: str = "all"
) -> Image.Image:
    """Construct the binary inpaint generation mask for SDXL (255 = inpaint, 0 = keep).
    
    - 'scene': Subject is kept (0), background is inpainted (255).
    - 'model': Kept garment is protected (0), new model body/face/limbs & scene are inpainted (255).
    """
    matte_np = np.array(subject_matte.convert("L"))
    h, w = matte_np.shape

    if mode == "scene":
        # Keep subject (value 0), inpaint background (value 255)
        # Dilate the inpaint area slightly into the subject edges (2px) so boundary blends
        inpaint_np = 255 - matte_np
        inpaint_np = cv2.dilate(inpaint_np, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        return Image.fromarray(inpaint_np)

    elif mode == "model":
        # Model mode: Generate a new human model and new scene around the existing garment
        # The kept garment is protected (0), everything else is inpainted (255)
        if garment_mask is not None:
            g_np = np.array(garment_mask.convert("L"))
            # Protected region is where garment mask is strong (> 128)
            # Inpaint area is 255 everywhere except protected garment
            inpaint_np = np.full((h, w), 255, dtype=np.uint8)
            inpaint_np[g_np > 128] = 0
            
            # Dilate the inpaint region slightly so diffusion generates seamlessly under the kept garment edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            inpaint_np = cv2.dilate(inpaint_np, kernel)
            return Image.fromarray(inpaint_np)
        else:
            # Fallback if no specific garment mask: keep inner 70% of subject matte
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            eroded_matte = cv2.erode(matte_np, kernel)
            inpaint_np = np.full((h, w), 255, dtype=np.uint8)
            inpaint_np[eroded_matte > 128] = 0
            return Image.fromarray(inpaint_np)

    # Default: inpaint everything
    return Image.new("L", (w, h), 255)


def generate_contact_shadow(
    subject_matte: Image.Image,
    target_size: Tuple[int, int],
    blur_radius: int = 18,
    opacity: float = 0.22,
    offset_y: int = 12
) -> Image.Image:
    """Generate a soft, realistic contact shadow on the floor for packshot or studio modes."""
    matte_np = np.array(subject_matte.convert("L"))
    h, w = matte_np.shape
    tw, th = target_size

    # Find bottom contact edge of subject
    y_indices, x_indices = np.where(matte_np > 50)
    if len(y_indices) == 0:
        return Image.new("RGBA", (tw, th), (0, 0, 0, 0))

    bottom_y = int(np.max(y_indices))
    min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))
    span_w = max_x - min_x

    # Create elliptical shadow patch under the base
    shadow_patch_h = max(20, int(span_w * 0.15))
    shadow_patch_w = int(span_w * 1.05)
    
    shadow_img = np.zeros((th, tw), dtype=np.uint8)
    center_x = (min_x + max_x) // 2
    center_y = min(th - 5, bottom_y + offset_y)
    
    cv2.ellipse(
        shadow_img,
        (center_x, center_y),
        (shadow_patch_w // 2, shadow_patch_h // 2),
        0, 0, 360,
        255,
        -1
    )

    # Gaussian blur for soft shadow falloff
    blur_k = blur_radius * 2 + 1
    blurred = cv2.GaussianBlur(shadow_img, (blur_k, blur_k), 0)
    
    # Scale intensity by opacity
    shadow_alpha = (blurred.astype(np.float32) * opacity).astype(np.uint8)
    
    # Create black shadow with computed alpha
    shadow_rgba = np.zeros((th, tw, 4), dtype=np.uint8)
    shadow_rgba[:, :, 3] = shadow_alpha
    
    return Image.fromarray(shadow_rgba, mode="RGBA")
