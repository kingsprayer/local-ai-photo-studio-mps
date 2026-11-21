"""SDXL dual-encoder prompt composition and token length management."""

import os
from typing import Dict, Any, Tuple, List, Optional
import yaml

DEFAULT_STYLE_GUIDE_PATH = "style_guide.yaml"

# Approximate subword token estimation if transformers CLIPTokenizer is not loaded in memory
def estimate_token_count(text: str) -> int:
    """Estimate CLIP BPE token count for prompt safety monitoring."""
    if not text:
        return 0
    words = text.strip().split()
    count = 2  # BOS and EOS tokens
    for w in words:
        # Punctuation or special characters often split
        punct_count = sum(1 for c in w if c in ",.:;!?-_/()[]'\"")
        length_penalty = max(1, len(w) // 4)
        count += punct_count + length_penalty
    return count


def load_style_guide(path: str = DEFAULT_STYLE_GUIDE_PATH) -> Dict[str, Any]:
    """Dynamically reload style_guide.yaml on every job execution."""
    if not os.path.exists(path):
        # Fallback to default in directory above if executed from subfolder
        alt_path = os.path.join(os.path.dirname(__file__), "..", path)
        if os.path.exists(alt_path):
            path = alt_path
        else:
            return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_prompts(
    attributes: Dict[str, Any],
    mode: str,  # 'packshot', 'scene', 'model'
    scene_key: Optional[str] = "minimal_studio",
    custom_scene: Optional[str] = None,
    gender: str = "women",
    pose_key: Optional[str] = "stand_relaxed",
    background_mode: str = "new",  # 'new', 'exact', 'inspiration'
    background_desc: Optional[str] = None,
    style_guide_path: str = DEFAULT_STYLE_GUIDE_PATH
) -> Dict[str, Any]:
    """Build SDXL dual text encoder prompts with strict 77-token budgeting.
    
    Prompt 1 (Text Encoder 1): Subject, garment details, scene/setting context.
    Prompt 2 (Text Encoder 2): Lighting, camera, aesthetic, styling & mood.
    """
    guide = load_style_guide(style_guide_path)
    
    # 1. Garment Description
    cat = attributes.get("category", "apparel")
    col = attributes.get("colour", "")
    fab = attributes.get("fabric", "")
    pat = attributes.get("pattern", "")
    fit = attributes.get("fit", "")
    det = attributes.get("details", "")
    
    garment_parts = [p for p in [col, pat, fit, fab, cat] if p and p.lower() != "none" and p.lower() != "solid"]
    garment_phrase = " ".join(garment_parts) if garment_parts else "apparel garment"
    if det and det.lower() not in ["none", "minimal", ""]:
        garment_phrase += f" with {det}"

    # 2. Subject & Model
    models_cfg = guide.get("models", {})
    model_info = models_cfg.get(gender, {})
    model_desc = model_info.get("description", "Indian fashion model with natural skin texture and poised expression")
    
    poses_cfg = guide.get("poses", {})
    pose_desc = poses_cfg.get(pose_key, "relaxed standing posture")

    # 3. Scene Context
    named_scenes = guide.get("named_scenes", {})
    scene_data = named_scenes.get(scene_key, {})
    scene_prompt = custom_scene if custom_scene else scene_data.get("prompt", "minimalist luxury fashion studio with warm limestone cyclorama")
    lighting_prompt = scene_data.get("lighting_prompt", guide.get("lighting", {}).get("primary", "soft diffused daylight"))

    if background_mode == "inspiration" and background_desc:
        scene_prompt = f"inspired by {background_desc}, clean minimalist luxury aesthetic"
        lighting_prompt = f"natural ambient light matching {background_desc}"

    # 4. Construct Prompt 1 (Subject + Garment + Scene)
    if mode == "packshot":
        p1 = f"commercial packshot product photography of {garment_phrase}, centered, pure solid white background RGB 255"
        p2 = "even bright studio product lighting, sharp crisp fabric texture, zero shadows, commercial catalog grade"
    elif mode == "scene":
        p1 = f"catalog photo of {garment_phrase}, {scene_prompt}"
        p2 = f"{lighting_prompt}, 85mm prime lens, Hasselblad medium format color depth, elegant luxury aesthetic"
    elif mode == "model":
        p1 = f"full length portrait of {model_desc}, wearing {garment_phrase}, {pose_desc}, {scene_prompt}"
        p2 = f"{lighting_prompt}, 85mm f/2.8 lens, natural skin pores, tack-sharp garment weave, SAADAA brand aesthetic"
    else:
        p1 = f"{garment_phrase}, {scene_prompt}"
        p2 = f"{lighting_prompt}, high fashion editorial"

    # Clean and trim whitespace
    p1 = " ".join(p1.split())
    p2 = " ".join(p2.split())

    # Negative Prompt
    negative_prompt = guide.get("negative_prompt", (
        "ugly, deformed, bad anatomy, bad hands, missing fingers, extra limbs, "
        "blurry, low quality, cartoon, 3d render, watermark, text"
    ))

    # Token Count Evaluation & Warnings
    t1_count = estimate_token_count(p1)
    t2_count = estimate_token_count(p2)

    warnings: List[str] = []
    if t1_count > 75:
        warnings.append(f"Prompt 1 exceeds 77 token safety budget ({t1_count} tokens). Lower priority keywords may be truncated by SDXL.")
    if t2_count > 75:
        warnings.append(f"Prompt 2 exceeds 77 token safety budget ({t2_count} tokens). Lower priority lighting/style tokens may be truncated.")

    return {
        "prompt": p1,
        "prompt_2": p2,
        "negative_prompt": negative_prompt,
        "token_counts": {
            "prompt": t1_count,
            "prompt_2": t2_count
        },
        "warnings": warnings,
        "style_guide_brand": guide.get("brand", {}).get("name", "SAADAA")
    }
