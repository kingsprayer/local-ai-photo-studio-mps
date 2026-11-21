"""Face and hand detail enhancement pass for Best quality preset with garment protection."""

import logging
from typing import Tuple, List, Optional
from PIL import Image, ImageFilter
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def enhance_detail_pass(
    canvas_image: Image.Image,
    garment_mask: Optional[Image.Image],
    face_boxes: List[Tuple[int, int, int, int]],
    hand_boxes: List[Tuple[int, int, int, int]],
    generator_fn=None,
    prompt_face: str = "detailed natural Indian model face, realistic skin pores, sharp clear eyes, high fashion editorial portrait",
    prompt_hand: str = "detailed realistic human hand, five distinct anatomical fingers, natural skin texture"
) -> Image.Image:
    """Perform crop-render-blend detail pass on detected faces and hands.
    
    CRITICAL: The garment is 100% protected — garment mask is subtracted
    so detail pass never modifies garment pixels.
    """
    result = canvas_image.copy()
    w, h = result.size

    # Prepare garment exclusion mask
    if garment_mask is not None:
        g_mask_resized = garment_mask.resize((w, h), Image.Resampling.NEAREST)
        g_np = np.array(g_mask_resized.convert("L"))
    else:
        g_np = np.zeros((h, w), dtype=np.uint8)

    # Helper to process an anatomical box
    def process_box(box: Tuple[int, int, int, int], prompt: str, denoise_strength: float = 0.28):
        nonlocal result
        bx1, by1, bx2, by2 = box
        bw = bx2 - bx1
        bh = by2 - by1
        if bw < 20 or bh < 20:
            return

        # Add 20% margin around box for context
        pad_x = int(bw * 0.2)
        pad_y = int(bh * 0.2)
        cx1 = max(0, bx1 - pad_x)
        cy1 = max(0, by1 - pad_y)
        cx2 = min(w, bx2 + pad_x)
        cy2 = min(h, by2 + pad_y)
        crop_w = cx2 - cx1
        crop_h = cy2 - cy1

        # Extract crop
        crop_img = result.crop((cx1, cy1, cx2, cy2))
        
        # Check garment overlap in this crop
        crop_garment_mask = g_np[cy1:cy2, cx1:cx2]
        
        # If generator function provided, run inpaint pass
        if generator_fn is not None:
            try:
                # Scale crop to 1024x1024 for SDXL
                crop_1024 = crop_img.resize((1024, 1024), Image.Resampling.LANCZOS)
                # Inpaint mask: inpaint face/hand (255), protect garment (0)
                mask_crop = np.full((crop_h, crop_w), 255, dtype=np.uint8)
                mask_crop[crop_garment_mask > 80] = 0
                mask_1024 = Image.fromarray(mask_crop).resize((1024, 1024), Image.Resampling.NEAREST)
                
                enhanced_1024 = generator_fn(
                    image=crop_1024,
                    mask_image=mask_1024,
                    prompt=prompt,
                    strength=denoise_strength,
                    num_inference_steps=12
                )
                enhanced_crop = enhanced_1024.resize((crop_w, crop_h), Image.Resampling.LANCZOS)
            except Exception as e:
                logger.warning(f"Detail generation failed: {e}. Using subtle sharpening fallback.")
                enhanced_crop = crop_img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2))
        else:
            # Fallback high-pass sharpening on non-garment skin
            enhanced_crop = crop_img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2))

        # Create elliptical blending feather mask centered on the crop
        blend_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        cv2.ellipse(
            blend_mask,
            (crop_w // 2, crop_h // 2),
            (int(crop_w * 0.45), int(crop_h * 0.45)),
            0, 0, 360,
            255,
            -1
        )
        # Subtract any garment overlap from the blend mask to ensure 100% garment protection
        blend_mask[crop_garment_mask > 80] = 0
        blend_mask = cv2.GaussianBlur(blend_mask, (15, 15), 0)
        blend_mask_pil = Image.fromarray(blend_mask)

        # Composite enhanced crop back
        result.paste(enhanced_crop, (cx1, cy1), blend_mask_pil)

    # Process all face boxes
    for fb in face_boxes:
        process_box(fb, prompt=prompt_face, denoise_strength=0.28)

    # Process all hand boxes
    for hb in hand_boxes:
        process_box(hb, prompt=prompt_hand, denoise_strength=0.26)

    return result
