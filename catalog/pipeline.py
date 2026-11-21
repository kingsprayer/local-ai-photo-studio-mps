"""Master pipeline orchestrator for SAADAA catalog generation."""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from PIL import Image
import numpy as np

from catalog.channels import (
    get_channel,
    get_generation_dimensions,
    smart_crop_or_outpaint,
    CHANNEL_FORMATS
)
from catalog.compliance import check_compliance
from catalog.prompts import build_prompts, load_style_guide
from catalog.vision import extract_attributes, describe_background, evaluate_qa
from catalog.masks import (
    extract_subject_matte,
    create_inpaint_mask,
    trim_mask,
    feather_mask,
    generate_contact_shadow
)
from catalog.detect import (
    extract_garment_mask,
    detect_face_and_hand_regions
)
from catalog.pose import (
    render_openpose_skeleton,
    generate_preset_pose
)
from catalog.compose import (
    calculate_pixel_fidelity,
    composite_generation_mask,
    paste_kept_garment_last,
    create_packshot,
    composite_exact_background
)
from catalog.upscale import upscale_to_channel_size
from catalog.generator import run_generation
from catalog.detail import enhance_detail_pass

logger = logging.getLogger(__name__)

OUTPUTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "outputs")


def generate_sku_folder_name(job_name: str, attributes: Dict[str, Any]) -> Tuple[str, str]:
    """Derive folder and SKU from job name or attributes."""
    clean_name = os.path.splitext(os.path.basename(job_name))[0]
    folder = os.path.dirname(job_name) if "/" in job_name else "catalog"
    if not folder:
        folder = "catalog"
    sku = attributes.get("category", "apparel").lower() + "_" + clean_name[:12]
    sku = "".join(c if c.isalnum() or c in "_-" else "_" for c in sku)
    return folder, sku


def process_catalog_job(
    image_input: Image.Image,
    job_name: str,
    mode: str = "packshot",  # 'packshot', 'scene', 'model'
    channels: Optional[List[str]] = None,
    attributes: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
    mock_mode: bool = False
) -> Dict[str, Any]:
    """Execute the complete SAADAA AI catalog workflow for a product image.
    
    Coordinates attributes, matting, detection, pose, diffusion, compositing,
    channel adaptations, compliance checking, and library indexing.
    """
    def report(pct: int, step: str, msg: str):
        if progress_callback:
            progress_callback(pct, step, msg)
        logger.info(f"[{pct}%] {step}: {msg}")

    options = options or {}
    channels = channels or ["amazon", "myntra", "shopify"]
    primary_channel = channels[0] if channels else "amazon"
    quality_preset = options.get("quality_preset", "standard")  # 'draft', 'standard', 'best'
    keep_mode = options.get("keep_mode", "all")  # 'upper', 'lower', 'dress', 'all', '+shoes'
    background_mode = options.get("background_mode", "new")  # 'new', 'exact', 'inspiration'
    bg_image = options.get("background_image")
    pose_keypoints = options.get("pose_keypoints")
    pose_preset = options.get("pose_preset")
    scene_key = options.get("scene_key", "minimal_studio")
    custom_scene = options.get("custom_scene")
    gender = options.get("gender", "women")
    seed = options.get("seed")

    report(5, "Attributes", "Analyzing apparel product attributes...")
    # 1. Attributes
    if not attributes:
        attributes = extract_attributes(image_input)

    report(15, "Matting", "Extracting high-precision BiRefNet subject matte...")
    # 2. Subject Matte (BiRefNet via rembg)
    subject_matte = extract_subject_matte(image_input)

    # 3. Determine base generation resolution for primary channel
    base_gen_w, base_gen_h = get_generation_dimensions(primary_channel)
    report(20, "Composition", f"Base generation resolution set to {base_gen_w}x{base_gen_h} for {primary_channel}.")

    # Intermediate master result
    master_result: Image.Image
    kept_mask_for_fidelity: Image.Image
    garment_mask: Optional[Image.Image] = None

    # MODE 1: PACKSHOT
    if mode == "packshot":
        report(40, "Packshot", "Generating deterministic pure-white catalog packshot with contact shadow...")
        fill_pct = options.get("fill_percent", 85)
        packshot_canvas, product_mask = create_packshot(
            source_image=image_input,
            subject_matte=subject_matte,
            target_size=(base_gen_w, base_gen_h),
            fill_percent=fill_pct,
            bg_rgb=(255, 255, 255),
            add_shadow=True
        )
        master_result = packshot_canvas
        kept_mask_for_fidelity = product_mask

    # MODE 2 & 3: SCENE & MODEL
    else:
        # Resize input to base generation size
        init_image = image_input.resize((base_gen_w, base_gen_h), Image.Resampling.LANCZOS)
        resized_matte = subject_matte.resize((base_gen_w, base_gen_h), Image.Resampling.BILINEAR)

        # Background description if inspiration mode
        bg_desc = None
        if background_mode == "inspiration" and bg_image is not None:
            report(25, "Vision Analysis", "Analyzing inspiration background architecture and lighting...")
            if mock_mode:
                bg_desc = "minimalist architectural space with warm lime-wash walls and polished microcement"
            else:
                bg_desc = describe_background(bg_image)

        # Build Prompts with strict SDXL 77-token budgeting
        prompt_data = build_prompts(
            attributes=attributes,
            mode=mode,
            scene_key=scene_key,
            custom_scene=custom_scene,
            gender=gender,
            pose_key=pose_preset or "stand_relaxed",
            background_mode=background_mode,
            background_desc=bg_desc
        )

        # Garment detection for Model mode
        if mode == "model":
            report(30, "Garment Segmentation", f"Segmenting garment mask (keep: {keep_mode}) with Florence-2 + SAM 2.1...")
            garment_mask = extract_garment_mask(
                image=init_image,
                person_matte=resized_matte,
                keep_mode=keep_mode,
                category_hint=attributes.get("category")
            )
        else:
            garment_mask = resized_matte

        # Construct generation inpaint mask
        inpaint_mask = create_inpaint_mask(
            mode=mode,
            subject_matte=resized_matte,
            garment_mask=garment_mask,
            keep_mode=keep_mode
        )

        # Pose conditioning
        pose_conditioning_map = None
        if pose_keypoints:
            report(35, "Pose Conditioning", "Rendering 18-point OpenPose skeleton from 3D studio...")
            pose_conditioning_map = render_openpose_skeleton(
                pose_keypoints,
                canvas_size=(base_gen_w, base_gen_h),
                face_visible=options.get("face_visible", True)
            )
        elif pose_preset:
            report(35, "Pose Conditioning", f"Rendering preset pose skeleton: {pose_preset}...")
            preset_points = generate_preset_pose(pose_preset, width=base_gen_w, height=base_gen_h)
            pose_conditioning_map = render_openpose_skeleton(
                preset_points,
                canvas_size=(base_gen_w, base_gen_h),
                face_visible=True
            )

        # Background conditioning if exact or inspiration
        strength = 1.0
        if background_mode == "exact" and bg_image is not None:
            # Composite model on exact background
            init_bg = bg_image.resize((base_gen_w, base_gen_h), Image.Resampling.LANCZOS)
            # Use original background as canvas
            init_image = Image.composite(init_image, init_bg, resized_matte)
            strength = 0.95
        elif background_mode == "inspiration" and bg_image is not None:
            init_bg = bg_image.resize((base_gen_w, base_gen_h), Image.Resampling.LANCZOS)
            init_image = Image.composite(init_image, init_bg, resized_matte)
            strength = float(options.get("inspiration_strength", 0.75))

        # Run diffusion generation
        report(50, "Diffusion", f"Running SDXL inpainting ({quality_preset} preset, strength={strength})...")
        generated_diffusion = run_generation(
            init_image=init_image,
            mask_image=inpaint_mask,
            prompt=prompt_data["prompt"],
            prompt_2=prompt_data["prompt_2"],
            negative_prompt=prompt_data["negative_prompt"],
            pose_image=pose_conditioning_map,
            quality_preset=quality_preset,
            strength=strength,
            seed=seed,
            mock_mode=mock_mode
        )

        # Best quality detail pass (face and hands)
        if quality_preset == "best":
            report(65, "Detail Enhancement", "Executing high-res face and hand detail pass with garment protection...")
            anatomical_regions = detect_face_and_hand_regions(generated_diffusion, resized_matte)
            generated_diffusion = enhance_detail_pass(
                canvas_image=generated_diffusion,
                garment_mask=garment_mask,
                face_boxes=anatomical_regions["faces"],
                hand_boxes=anatomical_regions["hands"]
            )

        # Strict Compositing
        report(75, "Compositing", "Executing deterministic pixel compositing and 100% garment protection...")
        if background_mode == "exact" and bg_image is not None:
            master_result = composite_exact_background(
                background_photo=bg_image.resize((base_gen_w, base_gen_h), Image.Resampling.LANCZOS),
                source_garment_photo=init_image,
                generated_model=generated_diffusion,
                person_matte=resized_matte,
                kept_garment_mask=garment_mask
            )
        else:
            # Composite through generation mask so everything outside it stays pixel-identical
            base_composite = composite_generation_mask(init_image, generated_diffusion, inpaint_mask)
            # Paste kept garment back last (trimmed 2px to remove old background fringe)
            master_result = paste_kept_garment_last(
                composite_canvas=base_composite,
                source_garment_image=init_image,
                garment_mask=garment_mask,
                trim_px=2
            )

        kept_mask_for_fidelity = garment_mask

    # Calculate exact pixel fidelity of kept region
    if mode == "packshot":
        pixel_fidelity = 100.0
    else:
        pixel_fidelity = calculate_pixel_fidelity(init_image, master_result, kept_mask_for_fidelity)
    report(82, "Fidelity Audit", f"Kept garment pixel fidelity: {pixel_fidelity}%")

    # Adapt across all chosen channels
    report(85, "Channel Formats", f"Exporting {len(channels)} marketplace channel formats...")
    folder_name, sku = generate_sku_folder_name(job_name, attributes)
    sku_dir = os.path.join(OUTPUTS_ROOT, folder_name, sku, mode)
    os.makedirs(sku_dir, exist_ok=True)

    channel_outputs = {}
    channel_compliance = {}

    for ch_id in channels:
        ch_spec = get_channel(ch_id)
        target_w = ch_spec["width"]
        target_h = ch_spec["height"]

        # Smart crop or outpaint
        adapted = smart_crop_or_outpaint(
            image=master_result,
            subject_mask=kept_mask_for_fidelity,
            target_w=target_w,
            target_h=target_h
        )

        # Upscale with Real-ESRGAN (tiled) if needed
        final_channel_img = upscale_to_channel_size(
            image=adapted,
            target_size=(target_w, target_h),
            quality_preset=quality_preset
        )

        # Save JPEG
        out_filename = f"{ch_id}.jpg"
        out_filepath = os.path.join(sku_dir, out_filename)
        final_channel_img.save(out_filepath, format="JPEG", quality=95, optimize=True)

        # Save cached WebP thumbnail for fast gallery rendering
        thumb_filename = f"{ch_id}_thumb.webp"
        thumb_filepath = os.path.join(sku_dir, thumb_filename)
        thumb_img = final_channel_img.copy()
        thumb_img.thumbnail((480, 480))
        thumb_img.save(thumb_filepath, format="WEBP", quality=85)

        # Run compliance audit
        compliance_result = check_compliance(final_channel_img, ch_id, file_path=out_filepath)
        
        channel_outputs[ch_id] = {
            "channel_id": ch_id,
            "channel_name": ch_spec["name"],
            "file_path": out_filepath,
            "relative_url": f"/outputs/{folder_name}/{sku}/{mode}/{out_filename}",
            "thumb_url": f"/outputs/{folder_name}/{sku}/{mode}/{thumb_filename}",
            "dimensions": [target_w, target_h],
            "compliance": compliance_result
        }
        channel_compliance[ch_id] = compliance_result

    # 4. QA Evaluation via Vision Model
    report(95, "Quality Assurance", "Auditing garment match, anatomy, realism, and background...")
    if mock_mode:
        qa_results = {
            "garment_match": 98.0,
            "anatomy": 95.0,
            "realism": 96.0,
            "background": 97.0,
            "overall_score": 96.5,
            "pixel_fidelity": pixel_fidelity,
            "verdict": "PASS",
            "reason": "Smoke test validation: garment match and anatomy are pristine."
        }
    else:
        qa_results = evaluate_qa(
            source_image=image_input,
            generated_image=master_result,
            pixel_fidelity_score=pixel_fidelity
        )

    # 5. Metadata JSON (sanitize non-serializable objects like PIL Images)
    clean_options = {}
    for ok, ov in options.items():
        if isinstance(ov, Image.Image):
            clean_options[ok] = f"<PIL.Image {ov.size} {ov.mode}>"
        else:
            clean_options[ok] = ov

    meta = {
        "job_name": job_name,
        "folder": folder_name,
        "sku": sku,
        "mode": mode,
        "created_at": time.time(),
        "attributes": attributes,
        "options": clean_options,
        "quality_preset": quality_preset,
        "pixel_fidelity": pixel_fidelity,
        "qa": qa_results,
        "channels": channel_outputs,
        "tags": [attributes.get("category", "apparel"), attributes.get("colour", ""), mode]
    }
    meta_path = os.path.join(sku_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    report(100, "Complete", f"Catalog assets generated with {pixel_fidelity}% garment fidelity.")

    return {
        "job_name": job_name,
        "folder": folder_name,
        "sku": sku,
        "mode": mode,
        "meta_path": meta_path,
        "pixel_fidelity": pixel_fidelity,
        "qa": qa_results,
        "channels": channel_outputs
    }
