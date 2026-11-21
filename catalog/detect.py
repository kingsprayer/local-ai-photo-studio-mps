"""Garment, face, and hand detection using Florence-2-base (MIT) boxes and SAM 2.1 (Apache-2.0) masks.

Strictly adheres to commercial-use licensing:
NO SegFormer clothes parsers (NVIDIA non-commercial license prohibited).
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import os
import torch
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

# Global cached detector models
_FLORENCE_MODEL = None
_FLORENCE_PROCESSOR = None
_SAM_MODEL = None
_SAM_PROCESSOR = None

# Short nouns for Florence-2 grounding
TARGET_NOUNS = {
    "upper": ["top", "shirt", "kurta", "jacket", "blouse"],
    "lower": ["pants", "trousers", "skirt", "bottoms"],
    "dress": ["dress", "gown", "one-piece", "kurta set"],
    "shoes": ["shoes", "sandals", "footwear"],
    "face": ["human face", "face"],
    "hand": ["hand", "hands"]
}


def get_device() -> torch.device:
    """Get the optimal PyTorch compute device (MPS on Apple Silicon, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_florence_if_available():
    """Load Florence-2-base if cached or accessible."""
    global _FLORENCE_MODEL, _FLORENCE_PROCESSOR
    if os.environ.get("SAADAA_MOCK_DIFFUSION") == "1":
        return None, None
    if _FLORENCE_MODEL is not None:
        return _FLORENCE_MODEL, _FLORENCE_PROCESSOR

    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
        model_id = "microsoft/Florence-2-base"
        device = get_device()
        logger.info(f"Loading Florence-2-base on {device}...")
        _FLORENCE_PROCESSOR = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        _FLORENCE_MODEL = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.type == "mps" else torch.float32,
            trust_remote_code=True
        ).to(device).eval()
        return _FLORENCE_MODEL, _FLORENCE_PROCESSOR
    except Exception as e:
        logger.info(f"Florence-2-base not loaded ({e}). Using proportional/color fallback.")
        return None, None


def load_sam_if_available():
    """Load SAM 2.1 hiera-small if available."""
    global _SAM_MODEL, _SAM_PROCESSOR
    if os.environ.get("SAADAA_MOCK_DIFFUSION") == "1":
        return None, None
    if _SAM_MODEL is not None:
        return _SAM_MODEL, _SAM_PROCESSOR

    try:
        from transformers import Sam2Model, Sam2Processor
        model_id = "facebook/sam2.1-hiera-small"
        device = get_device()
        logger.info(f"Loading SAM 2.1 hiera-small on {device}...")
        _SAM_PROCESSOR = Sam2Processor.from_pretrained(model_id)
        _SAM_MODEL = Sam2Model.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.type == "mps" else torch.float32
        ).to(device).eval()
        return _SAM_MODEL, _SAM_PROCESSOR
    except Exception as e:
        logger.info(f"SAM 2.1 not loaded ({e}). Using proportional/contour fallback.")
        return None, None


def detect_boxes_florence(image: Image.Image, text_queries: List[str]) -> List[Tuple[int, int, int, int]]:
    """Detect bounding boxes using Florence-2 open vocabulary grounding with short nouns."""
    model, processor = load_florence_if_available()
    if model is None or processor is None:
        return []

    boxes = []
    device = get_device()
    prompt_task = "<CAPTION_TO_PHRASE_GROUNDING>"
    
    for query in text_queries:
        full_prompt = f"{prompt_task} {query}"
        inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3
            )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            generated_text,
            task=prompt_task,
            image_size=(image.width, image.height)
        )
        grounding = parsed.get(prompt_task, {})
        for b in grounding.get("bboxes", []):
            boxes.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))

    return boxes


def segment_boxes_sam(image: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> Optional[Image.Image]:
    """Generate precise segmentation masks for bounding boxes using SAM 2.1."""
    if not boxes:
        return None
    model, processor = load_sam_if_available()
    if model is None or processor is None:
        return None

    try:
        device = get_device()
        inputs = processor(images=image, input_boxes=[boxes], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0]
        # Combine all masks with logical OR
        combined_mask = np.zeros((image.height, image.width), dtype=np.uint8)
        for m in masks:
            m_np = m[0].numpy().astype(np.uint8) * 255
            combined_mask = np.maximum(combined_mask, m_np)
        return Image.fromarray(combined_mask)
    except Exception as e:
        logger.warning(f"SAM 2.1 inference failed: {e}")
        return None


def get_proportional_fallback_mask(
    image: Image.Image,
    person_matte: Image.Image,
    target_part: str
) -> Image.Image:
    """Anatomical proportional fallback mask restricted to the BiRefNet person matte.
    
    Uses standard human anatomical proportions:
    - Head/Face: top 0% to 15% of person height.
    - Upper body (Top/Kurta/Shirt): 15% to 60% of person height.
    - Lower body (Pants/Trousers): 55% to 92% of person height.
    - Dress/Full set: 15% to 92% of person height.
    - Shoes: bottom 88% to 100% of person height.
    - Hands: 45% to 70% height, outer lateral edges.
    """
    matte_np = np.array(person_matte.convert("L"))
    h, w = matte_np.shape
    y_indices, x_indices = np.where(matte_np > 40)
    
    if len(y_indices) == 0:
        return Image.new("L", (w, h), 0)

    min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))
    person_h = max(10, max_y - min_y)
    
    part_mask = np.zeros((h, w), dtype=np.uint8)
    
    if target_part == "upper":
        y1 = int(min_y + person_h * 0.14)
        y2 = int(min_y + person_h * 0.62)
        part_mask[y1:y2, :] = 255
    elif target_part == "lower":
        y1 = int(min_y + person_h * 0.55)
        y2 = int(min_y + person_h * 0.94)
        part_mask[y1:y2, :] = 255
    elif target_part in ["dress", "all"]:
        y1 = int(min_y + person_h * 0.14)
        y2 = int(min_y + person_h * 0.94)
        part_mask[y1:y2, :] = 255
    elif target_part == "shoes":
        y1 = int(min_y + person_h * 0.88)
        y2 = max_y
        part_mask[y1:y2, :] = 255
    elif target_part == "face":
        y1 = min_y
        y2 = int(min_y + person_h * 0.18)
        part_mask[y1:y2, :] = 255
    elif target_part == "hand":
        y1 = int(min_y + person_h * 0.40)
        y2 = int(min_y + person_h * 0.75)
        part_mask[y1:y2, :] = 255
    else:
        part_mask[:, :] = 255

    # Strictly intersect with the BiRefNet person matte
    final_mask = cv2.bitwise_and(part_mask, matte_np)
    return Image.fromarray(final_mask)


def extract_garment_mask(
    image: Image.Image,
    person_matte: Image.Image,
    keep_mode: str = "all",  # 'upper', 'lower', 'dress', 'all', '+shoes'
    category_hint: Optional[str] = None
) -> Image.Image:
    """Extract garment mask using Florence-2 boxes + SAM 2.1 masks, restricted to BiRefNet person matte.
    
    Guarantees that SegFormer is NEVER used, preserving 100% commercial license compliance.
    """
    matte_np = np.array(person_matte.convert("L"))
    h, w = image.height, image.width

    # Determine query keywords based on keep mode and category hint
    text_queries = []
    if keep_mode == "upper":
        text_queries = TARGET_NOUNS["upper"]
    elif keep_mode == "lower":
        text_queries = TARGET_NOUNS["lower"]
    elif keep_mode == "dress":
        text_queries = TARGET_NOUNS["dress"]
    elif keep_mode in ["all", "everything"]:
        text_queries = TARGET_NOUNS["upper"] + TARGET_NOUNS["lower"] + TARGET_NOUNS["dress"]
    elif keep_mode == "+shoes":
        text_queries = TARGET_NOUNS["upper"] + TARGET_NOUNS["lower"] + TARGET_NOUNS["dress"] + TARGET_NOUNS["shoes"]
    else:
        text_queries = ["top", "pants", "dress"]

    # Try Florence-2 + SAM 2.1 pipeline
    boxes = detect_boxes_florence(image, text_queries)
    if boxes:
        sam_mask = segment_boxes_sam(image, boxes)
        if sam_mask:
            # Strictly restrict to person matte
            sam_np = np.array(sam_mask.convert("L"))
            restricted = cv2.bitwise_and(sam_np, matte_np)
            return Image.fromarray(restricted)

    # Use anatomical proportional fallback intersected with BiRefNet matte
    return get_proportional_fallback_mask(image, person_matte, target_part=keep_mode)


def detect_face_and_hand_regions(
    image: Image.Image,
    person_matte: Image.Image
) -> Dict[str, List[Tuple[int, int, int, int]]]:
    """Locate bounding boxes for face and hands for the Best quality detail enhancement pass."""
    face_boxes = detect_boxes_florence(image, TARGET_NOUNS["face"])
    hand_boxes = detect_boxes_florence(image, TARGET_NOUNS["hand"])
    
    # Fallback bounding boxes from person matte proportions if Florence isn't loaded
    matte_np = np.array(person_matte.convert("L"))
    y_indices, x_indices = np.where(matte_np > 50)
    
    if len(y_indices) > 0:
        min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))
        min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))
        person_h = max_y - min_y
        person_w = max_x - min_x
        center_x = (min_x + max_x) // 2

        if not face_boxes:
            # Head/face box
            fb_h = int(person_h * 0.18)
            fb_w = int(fb_h * 0.85)
            fx1 = max(0, center_x - fb_w // 2)
            fy1 = min_y
            fx2 = min(image.width, fx1 + fb_w)
            fy2 = min(image.height, fy1 + fb_h)
            face_boxes = [(fx1, fy1, fx2, fy2)]

        if not hand_boxes:
            # Left and right hand estimated regions
            hw = int(person_w * 0.22)
            hh = int(person_h * 0.15)
            hy = int(min_y + person_h * 0.52)
            
            # Left lateral
            hx1 = max(0, min_x - 10)
            hand_boxes.append((hx1, hy, min(image.width, hx1 + hw), min(image.height, hy + hh)))
            # Right lateral
            hx2 = min(image.width - hw, max_x - hw + 10)
            hand_boxes.append((max(0, hx2), hy, min(image.width, hx2 + hw), min(image.height, hy + hh)))

    return {
        "faces": face_boxes,
        "hands": hand_boxes
    }
