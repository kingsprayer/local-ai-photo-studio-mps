"""OpenPose 18-keypoint skeleton rendering and mannequin conditioning for ControlNet."""

from typing import Dict, Any, List, Tuple, Optional
import math
import numpy as np
from PIL import Image
import cv2

# Standard OpenPose 18-point limb connections and color palette
LIMB_CONNECTIONS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),  # Upper body & arms
    (1, 8), (8, 9), (9, 10), (10, 11),                # Torso to Right leg
    (8, 12), (12, 13), (13, 14),                      # Torso to Left leg
    (1, 0), (0, 15), (15, 17), (0, 16), (16, 18),    # Head, eyes, ears
    (2, 9), (5, 12)                                   # Clavicle/hips lateral
]

LIMB_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
    (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85),
    (128, 255, 0), (0, 255, 128)
]

KEYPOINT_NAMES = [
    "nose", "neck", "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist", "mid_hip", "r_hip",
    "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear"
]


def render_openpose_skeleton(
    keypoints_dict_or_list: Any,
    canvas_size: Tuple[int, int] = (1024, 1024),
    face_visible: bool = True
) -> Image.Image:
    """Render a standard 18-point OpenPose colored skeleton image for ControlNet."""
    w, h = canvas_size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Normalize keypoint list: either list of (x, y, conf) or dict with index/names
    points: List[Optional[Tuple[int, int]]] = [None] * 19

    if isinstance(keypoints_dict_or_list, list):
        for idx, item in enumerate(keypoints_dict_or_list):
            if idx >= 19:
                break
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                x, y = int(item[0]), int(item[1])
                conf = float(item[2]) if len(item) > 2 else 1.0
                if conf > 0.1 and 0 <= x < w and 0 <= y < h:
                    points[idx] = (x, y)
    elif isinstance(keypoints_dict_or_list, dict):
        for idx, name in enumerate(KEYPOINT_NAMES):
            if name in keypoints_dict_or_list:
                item = keypoints_dict_or_list[name]
                x, y = int(item["x"]), int(item["y"])
                conf = float(item.get("confidence", 1.0))
                if conf > 0.1 and 0 <= x < w and 0 <= y < h:
                    points[idx] = (x, y)

    # Hide face points if mannequin is turned away
    if not face_visible:
        for f_idx in [0, 15, 16, 17, 18]:
            points[f_idx] = None

    # Draw limbs
    for i, (k1, k2) in enumerate(LIMB_CONNECTIONS):
        if k1 < len(points) and k2 < len(points):
            p1, p2 = points[k1], points[k2]
            if p1 is not None and p2 is not None:
                color = LIMB_COLORS[i % len(LIMB_COLORS)]
                # Draw thick anti-aliased line
                cv2.line(canvas, p1, p2, color, thickness=6, lineType=cv2.LINE_AA)

    # Draw joints (colored circles)
    for idx, p in enumerate(points):
        if p is not None:
            color = LIMB_COLORS[idx % len(LIMB_COLORS)]
            cv2.circle(canvas, p, radius=5, color=color, thickness=-1, lineType=cv2.LINE_AA)

    return Image.fromarray(canvas)


def calculate_shoulder_geometry(
    keypoints: Any
) -> Dict[str, Any]:
    """Measure shoulder position, width, and tilt angle for the Snap to Body feature."""
    # Find r_shoulder (index 2) and l_shoulder (index 5)
    r_pt, l_pt = None, None
    if isinstance(keypoints, list) and len(keypoints) >= 6:
        r_pt = keypoints[2]
        l_pt = keypoints[5]
    elif isinstance(keypoints, dict):
        if "r_shoulder" in keypoints and "l_shoulder" in keypoints:
            r_pt = (keypoints["r_shoulder"]["x"], keypoints["r_shoulder"]["y"])
            l_pt = (keypoints["l_shoulder"]["x"], keypoints["l_shoulder"]["y"])

    if r_pt is None or l_pt is None:
        return {"valid": False, "center": (512, 300), "width": 250, "angle_deg": 0.0}

    rx, ry = float(r_pt[0]), float(r_pt[1])
    lx, ly = float(l_pt[0]), float(l_pt[1])

    cx = (rx + lx) / 2.0
    cy = (ry + ly) / 2.0
    dx = lx - rx
    dy = ly - ry
    width = math.sqrt(dx * dx + dy * dy)
    angle_deg = math.degrees(math.atan2(dy, dx))

    return {
        "valid": True,
        "center": (int(cx), int(cy)),
        "width": int(width),
        "angle_deg": round(angle_deg, 2)
    }


def generate_preset_pose(preset_name: str, width: int = 1024, height: int = 1024) -> List[Tuple[int, int, float]]:
    """Return default 18-keypoint OpenPose skeleton coordinates for built-in studio presets."""
    cx = width // 2
    cy = height // 2
    scale = height / 1024.0

    # Base relaxed standing points (relative to center)
    base = [
        (cx, int(cy - 360 * scale), 1.0),       # 0: Nose
        (cx, int(cy - 300 * scale), 1.0),       # 1: Neck
        (int(cx - 110 * scale), int(cy - 290 * scale), 1.0), # 2: R-Shoulder
        (int(cx - 140 * scale), int(cy - 120 * scale), 1.0), # 3: R-Elbow
        (int(cx - 150 * scale), int(cy + 50 * scale), 1.0),  # 4: R-Wrist
        (int(cx + 110 * scale), int(cy - 290 * scale), 1.0), # 5: L-Shoulder
        (int(cx + 140 * scale), int(cy - 120 * scale), 1.0), # 6: L-Elbow
        (int(cx + 150 * scale), int(cy + 50 * scale), 1.0),  # 7: L-Wrist
        (cx, int(cy - 60 * scale), 1.0),        # 8: Mid-Hip
        (int(cx - 80 * scale), int(cy - 50 * scale), 1.0),   # 9: R-Hip
        (int(cx - 85 * scale), int(cy + 170 * scale), 1.0),  # 10: R-Knee
        (int(cx - 90 * scale), int(cy + 400 * scale), 1.0),  # 11: R-Ankle
        (int(cx + 80 * scale), int(cy - 50 * scale), 1.0),   # 12: L-Hip
        (int(cx + 85 * scale), int(cy + 170 * scale), 1.0),  # 13: L-Knee
        (int(cx + 90 * scale), int(cy + 400 * scale), 1.0),  # 14: L-Ankle
        (int(cx - 18 * scale), int(cy - 375 * scale), 1.0),  # 15: R-Eye
        (int(cx + 18 * scale), int(cy - 375 * scale), 1.0),  # 16: L-Eye
        (int(cx - 50 * scale), int(cy - 365 * scale), 1.0),  # 17: R-Ear
        (int(cx + 50 * scale), int(cy - 365 * scale), 1.0),  # 18: L-Ear
    ]

    if preset_name == "hand_on_hip":
        # Right hand on hip
        base[3] = (int(cx - 160 * scale), int(cy - 180 * scale), 1.0)
        base[4] = (int(cx - 95 * scale), int(cy - 60 * scale), 1.0)
    elif preset_name == "hands_on_hips":
        base[3] = (int(cx - 160 * scale), int(cy - 180 * scale), 1.0)
        base[4] = (int(cx - 95 * scale), int(cy - 60 * scale), 1.0)
        base[6] = (int(cx + 160 * scale), int(cy - 180 * scale), 1.0)
        base[7] = (int(cx + 95 * scale), int(cy - 60 * scale), 1.0)
    elif preset_name == "walking":
        base[10] = (int(cx - 110 * scale), int(cy + 150 * scale), 1.0)
        base[11] = (int(cx - 130 * scale), int(cy + 380 * scale), 1.0)
        base[13] = (int(cx + 60 * scale), int(cy + 190 * scale), 1.0)
        base[14] = (int(cx + 40 * scale), int(cy + 410 * scale), 1.0)

    return base
