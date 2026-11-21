"""Marketplace compliance checker for exported catalog images."""

from typing import Dict, Any, Optional, Union
import os
from PIL import Image
import numpy as np
import cv2
from catalog.channels import get_channel


def calculate_sharpness(img_np: np.ndarray) -> float:
    """Calculate image sharpness using Laplacian variance."""
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def evaluate_border_whiteness(img_np: np.ndarray, border_pct: float = 0.04) -> Dict[str, Any]:
    """Check if the outer perimeter border meets pure white criteria (RGB >= 250)."""
    h, w = img_np.shape[:2]
    bw = max(2, int(w * border_pct))
    bh = max(2, int(h * border_pct))

    # Extract border stripes
    top = img_np[:bh, :]
    bottom = img_np[-bh:, :]
    left = img_np[:, :bw]
    right = img_np[:, -bw:]

    border_pixels = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ], axis=0)

    # Check RGB >= 250
    white_mask = (border_pixels[:, 0] >= 250) & (border_pixels[:, 1] >= 250) & (border_pixels[:, 2] >= 250)
    white_ratio = float(np.mean(white_mask)) * 100.0
    mean_rgb = [float(v) for v in np.mean(border_pixels, axis=0)]

    return {
        "white_ratio": round(white_ratio, 1),
        "mean_rgb": [round(c, 1) for c in mean_rgb],
        "is_pure_white": white_ratio >= 98.0
    }


def evaluate_product_fill(img_np: np.ndarray, threshold: int = 248) -> float:
    """Calculate the product fill percentage (non-white canvas area)."""
    # Non-white pixels are considered product/model pixels
    non_white = (img_np[:, :, 0] < threshold) | (img_np[:, :, 1] < threshold) | (img_np[:, :, 2] < threshold)
    fill_pct = float(np.mean(non_white)) * 100.0
    return round(fill_pct, 1)


def check_compliance(
    image_input: Union[str, Image.Image],
    channel_id: str,
    file_path: Optional[str] = None
) -> Dict[str, Any]:
    """Perform automated marketplace compliance audit on an image."""
    ch = get_channel(channel_id)
    
    if isinstance(image_input, str):
        file_path = image_input
        pil_img = Image.open(file_path).convert("RGB")
    else:
        pil_img = image_input.convert("RGB")
        
    img_np = np.array(pil_img)
    w, h = pil_img.size
    
    # 1. Exact Dimensions
    dim_match = (w == ch["width"] and h == ch["height"])
    dim_ratio = w / h
    ar_match = abs(dim_ratio - ch["aspect_ratio"]) < 0.01

    # 2. Min Resolution
    min_res_pass = (w >= min(ch["width"], ch["min_res"])) and (h >= min(ch["height"], ch["min_res"]))

    # 3. File Size
    file_size_bytes = 0
    file_size_mb = 0.0
    if file_path and os.path.exists(file_path):
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    size_pass = file_size_mb <= ch["max_size_mb"] if file_size_mb > 0 else True

    # 4. Sharpness
    sharpness = round(calculate_sharpness(img_np), 1)
    sharpness_pass = sharpness >= 200.0

    # 5. Border Whiteness & Product Fill (for Amazon/Google main images)
    border_info = None
    fill_pct = None
    border_pass = True
    fill_pass = True

    if ch.get("white_border_required"):
        border_info = evaluate_border_whiteness(img_np)
        border_pass = border_info["is_pure_white"]
        
        fill_pct = evaluate_product_fill(img_np)
        min_fill = ch.get("min_fill_pct", 80)
        fill_pass = fill_pct >= min_fill

    # Compile verdict
    critical_failures = []
    warnings = []

    if not dim_match:
        critical_failures.append(f"Dimension mismatch: got {w}x{h}, expected {ch['width']}x{ch['height']}")
    if not min_res_pass:
        critical_failures.append(f"Resolution below minimum {ch['min_res']}px")
    if not size_pass:
        critical_failures.append(f"File size {file_size_mb}MB exceeds limit of {ch['max_size_mb']}MB")
        
    if ch.get("white_border_required"):
        if not border_pass:
            warnings.append(f"Border not pure white (≥ RGB 250): {border_info['white_ratio']}% pure white")
        if not fill_pass:
            warnings.append(f"Product fill {fill_pct}% is below required {ch['min_fill_pct']}%")

    if not sharpness_pass:
        warnings.append(f"Sharpness score {sharpness} is below recommended 200 threshold")

    if len(critical_failures) > 0:
        status = "fail"
        badge = "✗"
    elif len(warnings) > 0:
        status = "warn"
        badge = "!"
    else:
        status = "pass"
        badge = "✓"

    summary_text = "All marketplace criteria passed." if status == "pass" else "; ".join(critical_failures + warnings)

    return {
        "channel_id": channel_id,
        "channel_name": ch["name"],
        "status": status,
        "badge": badge,
        "summary": summary_text,
        "dimensions": {
            "current": [w, h],
            "expected": [ch["width"], ch["height"]],
            "passed": dim_match
        },
        "sharpness": {
            "score": sharpness,
            "threshold": 200.0,
            "passed": sharpness_pass
        },
        "file_size_mb": file_size_mb,
        "max_size_mb": ch["max_size_mb"],
        "white_border": border_info,
        "product_fill_pct": fill_pct,
        "min_fill_required": ch.get("min_fill_pct"),
        "critical_failures": critical_failures,
        "warnings": warnings
    }
