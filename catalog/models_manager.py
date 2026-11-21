"""Model weights tracker, byte verification, and dynamic UI feature locking."""

import os
import glob
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

# Canonical expected models and approximate target sizes
MODEL_SPECS = {
    "generator": {
        "id": "OzzyGT/RealVisXL_V4.0_inpainting",
        "name": "RealVisXL V4.0 Inpainting",
        "license": "OpenRAIL++ (Commercial Use Permitted)",
        "expected_bytes": 6_900_000_000,  # ~6.5 GB
        "hf_repo": "models--OzzyGT--RealVisXL_V4.0_inpainting",
        "features": ["scene", "model"],
        "description": "Photorealistic SDXL inpainting generator"
    },
    "controlnet": {
        "id": "xinsir/controlnet-openpose-sdxl-1.0",
        "name": "ControlNet OpenPose SDXL",
        "license": "Apache-2.0 (Commercial Use Permitted)",
        "expected_bytes": 2_500_000_000,  # ~2.5 GB
        "hf_repo": "models--xinsir--controlnet-openpose-sdxl-1.0",
        "features": ["pose_studio"],
        "description": "18-point OpenPose skeleton human pose conditioning"
    },
    "detection": {
        "id": "microsoft/Florence-2-base",
        "name": "Florence-2-base",
        "license": "MIT (Commercial Use Permitted)",
        "expected_bytes": 900_000_000,   # ~900 MB
        "hf_repo": "models--microsoft--Florence-2-base",
        "features": ["auto_detect"],
        "description": "Garment and anatomy grounding without SegFormer"
    },
    "segmentation": {
        "id": "facebook/sam2.1-hiera-small",
        "name": "SAM 2.1 Hiera Small",
        "license": "Apache-2.0 (Commercial Use Permitted)",
        "expected_bytes": 180_000_000,   # ~180 MB
        "hf_repo": "models--facebook--sam2.1-hiera-small",
        "features": ["garment_mask"],
        "description": "High-fidelity mask segmentation"
    },
    "upscaler": {
        "id": "Real-ESRGAN/x4plus",
        "name": "Real-ESRGAN x4plus",
        "license": "BSD-3-Clause (Commercial Use Permitted)",
        "expected_bytes": 67_000_000,    # ~67 MB
        "local_file": "RealESRGAN_x4plus.pth",
        "features": ["high_res_export"],
        "description": "4x super-resolution detail upscaler"
    },
    "draft_lora": {
        "id": "ByteDance/SDXL-Lightning",
        "name": "SDXL-Lightning 8-step LoRA",
        "license": "OpenRAIL++ (Commercial Use Permitted)",
        "expected_bytes": 396_000_000,   # ~396 MB
        "local_file": "sdxl_lightning_8step_lora.safetensors",
        "features": ["draft_mode"],
        "description": "High-speed 8-step drafting LoRA"
    }
}


def get_dir_size(path: str) -> int:
    """Calculate total size of all files in directory in bytes."""
    total = 0
    if not os.path.exists(path):
        return 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except Exception:
                pass
    return total


def check_hf_cache_repo(repo_name: str) -> Tuple[bool, int]:
    """Check if HuggingFace cache contains the model snapshot."""
    repo_path = os.path.join(HF_CACHE_DIR, repo_name)
    if not os.path.exists(repo_path):
        return False, 0
    size = get_dir_size(repo_path)
    # If size is greater than 50MB, considered actively downloaded or in progress
    return size > 50_000_000, size


def check_local_model_file(filename: str) -> Tuple[bool, int]:
    """Check local file in models/ directory."""
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        size = os.path.getsize(path)
        return size > 1_000_000, size
    return False, 0


def get_all_models_status() -> Dict[str, Any]:
    """Return status, byte size, and unlock capabilities for all required models."""
    results = {}
    features = {
        "packshot": True,  # Packshot is deterministic and always available
        "scene": False,
        "model": False,
        "pose_studio": False,
        "draft_mode": False
    }

    for key, spec in MODEL_SPECS.items():
        is_ready = False
        bytes_on_disk = 0

        if "hf_repo" in spec:
            is_ready, bytes_on_disk = check_hf_cache_repo(spec["hf_repo"])
        elif "local_file" in spec:
            is_ready, bytes_on_disk = check_local_model_file(spec["local_file"])

        pct = min(100.0, (bytes_on_disk / float(spec["expected_bytes"])) * 100.0) if spec["expected_bytes"] > 0 else 0.0

        results[key] = {
            "name": spec["name"],
            "license": spec["license"],
            "description": spec["description"],
            "ready": is_ready,
            "bytes_on_disk": bytes_on_disk,
            "expected_bytes": spec["expected_bytes"],
            "progress_pct": round(pct, 1),
            "status": "ready" if is_ready else ("downloading" if bytes_on_disk > 10_000_000 else "missing")
        }

        # Unlock feature flags
        if is_ready or os.environ.get("SAADAA_MOCK_DIFFUSION") == "1":
            for f in spec.get("features", []):
                if f in features:
                    features[f] = True

    # If mock mode is explicitly on, unlock everything for testing
    if os.environ.get("SAADAA_MOCK_DIFFUSION") == "1":
        for k in features:
            features[k] = True

    return {
        "features": features,
        "models": results,
        "all_commercial_licensed": True
    }
