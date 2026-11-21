"""Channel definitions and smart crop/outpaint adaptations for e-commerce marketplaces."""

from typing import Dict, Any, Tuple, Optional
from PIL import Image, ImageFilter
import numpy as np
import cv2

CHANNEL_FORMATS: Dict[str, Dict[str, Any]] = {
    "amazon": {
        "id": "amazon",
        "name": "Amazon",
        "width": 2000,
        "height": 2000,
        "aspect_ratio": 1.0,
        "min_res": 1000,
        "max_size_mb": 10.0,
        "white_border_required": True,
        "min_fill_pct": 85,
        "format": "JPEG",
        "description": "Amazon Main Image: 2000x2000, Pure white background (RGB 255), >= 85% product fill."
    },
    "myntra": {
        "id": "myntra",
        "name": "Myntra",
        "width": 1080,
        "height": 1440,
        "aspect_ratio": 1080 / 1440,  # 0.75 (3:4)
        "min_res": 1080,
        "max_size_mb": 5.0,
        "white_border_required": False,
        "min_fill_pct": None,
        "format": "JPEG",
        "description": "Myntra Fashion: 1080x1440 (3:4), editorial portrait framing."
    },
    "shopify": {
        "id": "shopify",
        "name": "Shopify",
        "width": 1536,
        "height": 2048,
        "aspect_ratio": 1536 / 2048,  # 0.75 (3:4)
        "min_res": 1024,
        "max_size_mb": 20.0,
        "white_border_required": False,
        "min_fill_pct": None,
        "format": "JPEG",
        "description": "Shopify High-Res Storefront: 1536x2048 (3:4)."
    },
    "google": {
        "id": "google",
        "name": "Google Shopping",
        "width": 1500,
        "height": 1500,
        "aspect_ratio": 1.0,
        "min_res": 800,
        "max_size_mb": 10.0,
        "white_border_required": True,
        "min_fill_pct": 75,
        "format": "JPEG",
        "description": "Google Shopping: 1500x1500, Pure white background (>= RGB 250), >= 75% product fill."
    },
    "instagram_post": {
        "id": "instagram_post",
        "name": "Instagram Post",
        "width": 1080,
        "height": 1350,
        "aspect_ratio": 1080 / 1350,  # 0.8 (4:5)
        "min_res": 1080,
        "max_size_mb": 8.0,
        "white_border_required": False,
        "min_fill_pct": None,
        "format": "JPEG",
        "description": "Instagram Feed Portrait: 1080x1350 (4:5)."
    },
    "instagram_story": {
        "id": "instagram_story",
        "name": "Instagram Story / Reel",
        "width": 1080,
        "height": 1920,
        "aspect_ratio": 1080 / 1920,  # 0.5625 (9:16)
        "min_res": 1080,
        "max_size_mb": 10.0,
        "white_border_required": False,
        "min_fill_pct": None,
        "format": "JPEG",
        "description": "Instagram Story/Reel: 1080x1920 (9:16 vertical full bleed)."
    },
    "banner": {
        "id": "banner",
        "name": "Website Banner",
        "width": 1920,
        "height": 1080,
        "aspect_ratio": 1920 / 1080,  # 1.7778 (16:9)
        "min_res": 1080,
        "max_size_mb": 10.0,
        "white_border_required": False,
        "min_fill_pct": None,
        "format": "JPEG",
        "description": "Website Hero Banner: 1920x1080 (16:9 landscape)."
    }
}


def get_channel(channel_id: str) -> Dict[str, Any]:
    """Retrieve specifications for a given channel ID."""
    if channel_id not in CHANNEL_FORMATS:
        raise ValueError(f"Unknown channel: {channel_id}. Valid channels: {list(CHANNEL_FORMATS.keys())}")
    return CHANNEL_FORMATS[channel_id]


def get_generation_dimensions(primary_channel_id: str, max_base_dim: int = 1024) -> Tuple[int, int]:
    """Calculate the base SDXL generation dimensions (multiples of 64) for the primary channel.
    
    SDXL operates natively around 1024x1024 or equivalent 1 megapixel buckets
    (e.g., 896x1152, 768x1344, 1152x896).
    """
    ch = get_channel(primary_channel_id)
    target_ar = ch["aspect_ratio"]
    
    # Common SDXL native resolution buckets around 1024x1024 (1,048,576 pixels)
    sdxl_buckets = [
        (1024, 1024),  # 1:1 (1.0)
        (896, 1152),   # 3:4 / 4:5 (~0.78)
        (832, 1216),   # ~0.68
        (768, 1344),   # 9:16 (~0.57)
        (640, 1536),   # ~0.42
        (1152, 896),   # 4:3 (~1.29)
        (1216, 832),   # ~1.46
        (1344, 768),   # 16:9 (~1.75)
        (1536, 640)    # ~2.4
    ]
    
    # Select bucket that closest matches the channel aspect ratio
    best_bucket = min(sdxl_buckets, key=lambda b: abs((b[0] / b[1]) - target_ar))
    return best_bucket


def smart_crop_or_outpaint(
    image: Image.Image,
    subject_mask: Optional[Image.Image],
    target_w: int,
    target_h: int
) -> Image.Image:
    """Adapt a master image to target channel dimensions using subject-aware smart crop or outpainting.
    
    - If subject comfortably fits within target aspect ratio inside existing bounds: smart crop.
    - If target aspect ratio extends outside (e.g. 1:1 to 16:9 banner): outpaint background seamlessly.
    """
    img_w, img_h = image.size
    target_ar = target_w / target_h
    source_ar = img_w / img_h

    # Detect subject bounds from mask if provided, else use center 60%
    if subject_mask is not None:
        mask_np = np.array(subject_mask.convert("L"))
        y_indices, x_indices = np.where(mask_np > 30)
        if len(y_indices) > 0 and len(x_indices) > 0:
            sub_x1, sub_x2 = int(np.min(x_indices)), int(np.max(x_indices))
            sub_y1, sub_y2 = int(np.min(y_indices)), int(np.max(y_indices))
        else:
            sub_x1, sub_x2 = int(img_w * 0.2), int(img_w * 0.8)
            sub_y1, sub_y2 = int(img_h * 0.1), int(img_h * 0.9)
    else:
        sub_x1, sub_x2 = int(img_w * 0.2), int(img_w * 0.8)
        sub_y1, sub_y2 = int(img_h * 0.1), int(img_h * 0.9)

    sub_w = sub_x2 - sub_x1
    sub_h = sub_y2 - sub_y1
    sub_cx = (sub_x1 + sub_x2) / 2.0
    sub_cy = (sub_y1 + sub_y2) / 2.0

    # Case 1: Smart Crop (Image is wider or taller than target AR and subject fits)
    if abs(source_ar - target_ar) < 0.01:
        # Same aspect ratio, simply resize
        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # Calculate crop window in source image coordinates
    if source_ar > target_ar:
        # Source is wider than target: crop width
        crop_h = img_h
        crop_w = int(crop_h * target_ar)
        
        # Center horizontally on subject
        crop_x1 = int(sub_cx - crop_w / 2.0)
        crop_x1 = max(0, min(crop_x1, img_w - crop_w))
        crop_x2 = crop_x1 + crop_w
        
        # Check if subject fits horizontally
        if sub_x1 >= crop_x1 and sub_x2 <= crop_x2:
            cropped = image.crop((crop_x1, 0, crop_x2, crop_h))
            return cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)
    else:
        # Source is taller than target: crop height
        crop_w = img_w
        crop_h = int(crop_w / target_ar)
        
        # In fashion/portraits, prefer more headroom space (shift center slightly up)
        crop_y1 = int(sub_cy - crop_h * 0.45)
        crop_y1 = max(0, min(crop_y1, img_h - crop_h))
        crop_y2 = crop_y1 + crop_h
        
        # Check if subject fits vertically
        if sub_y1 >= crop_y1 and sub_y2 <= crop_y2:
            cropped = image.crop((0, crop_y1, crop_w, crop_y2))
            return cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # Case 2: Outpaint / Extended Canvas
    # When subject would be clipped by crop, scale subject to comfortably fit target canvas
    # and fill/extend background edges smoothly.
    scale = min(target_w / img_w, target_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    scaled_img = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create target canvas
    canvas = Image.new("RGB", (target_w, target_h), (248, 246, 240))
    pos_x = (target_w - new_w) // 2
    pos_y = (target_h - new_h) // 2

    # Synthesize background extension from the edges of scaled_img
    # Convert to cv2 for smooth background edge replication
    scaled_np = np.array(scaled_img)
    canvas_np = np.full((target_h, target_w, 3), 248, dtype=np.uint8)

    # Fill canvas with edge reflection blur
    bg_blurred = cv2.resize(scaled_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    bg_blurred = cv2.GaussianBlur(bg_blurred, (99, 99), 0)
    canvas_np = bg_blurred

    # Place pristine scaled image in center
    canvas_np[pos_y:pos_y + new_h, pos_x:pos_x + new_w] = scaled_np

    # Feather the boundary seam (3px) to prevent hard lines
    feather_w = min(12, pos_x) if pos_x > 0 else 0
    feather_h = min(12, pos_y) if pos_y > 0 else 0
    if feather_w > 0:
        for i in range(feather_w):
            alpha = (i + 1) / float(feather_w)
            canvas_np[pos_y:pos_y + new_h, pos_x + i] = (
                alpha * scaled_np[:, i] + (1 - alpha) * canvas_np[pos_y:pos_y + new_h, pos_x + i]
            ).astype(np.uint8)
            canvas_np[pos_y:pos_y + new_h, pos_x + new_w - 1 - i] = (
                alpha * scaled_np[:, new_w - 1 - i] + (1 - alpha) * canvas_np[pos_y:pos_y + new_h, pos_x + new_w - 1 - i]
            ).astype(np.uint8)

    return Image.fromarray(canvas_np)
