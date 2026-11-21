"""Gallery and Review Router for SAADAA Local Lifestyle AI Photography.

Provides endpoints for:
1. Viewing all 530 lifestyle looks across all 76 SKUs in all colors.
2. Liking / Disliking looks (approved images saved to saadaa_approved_lifestyle/,
   disliked images queued in regeneration_queue.json for re-generation).
3. 1-Click ZIP download of all approved lifestyle picks.
4. On-demand regeneration of individual disliked items or batch regeneration of queue.
"""

import os
import sys
import time
import json
import shutil
import zipfile
import io
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel

logger = logging.getLogger("saadaa.gallery")

router = APIRouter(prefix="/api/gallery", tags=["Gallery & Review"])

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
CATALOG_PATH = os.path.join(PROJECT_ROOT, "moodboard_data.json")
REVIEW_STATUS_PATH = os.path.join(PROJECT_ROOT, "lifestyle_review_status.json")
REGEN_QUEUE_PATH = os.path.join(PROJECT_ROOT, "regeneration_queue.json")
APPROVED_DIR = os.path.join(PROJECT_ROOT, "saadaa_approved_lifestyle")
BACKGROUNDS_DIR = os.path.join(PROJECT_ROOT, "background_plates")

os.makedirs(APPROVED_DIR, exist_ok=True)

# 8 SAADAA Architectural Background Plates for alternate compositions
BACKGROUND_PLATES = [
    os.path.join(BACKGROUNDS_DIR, "01_dhrangadhra_sandstone.jpg"),
    os.path.join(BACKGROUNDS_DIR, "02_chettinad_redoxide.jpg"),
    os.path.join(BACKGROUNDS_DIR, "03_kerala_verandah.jpg"),
    os.path.join(BACKGROUNDS_DIR, "04_mumbai_loft.jpg"),
    os.path.join(BACKGROUNDS_DIR, "05_goan_concrete_villa.jpg"),
    os.path.join(BACKGROUNDS_DIR, "06_nilgiris_verandah.jpg"),
    os.path.join(BACKGROUNDS_DIR, "07_jaipur_jali_terrace.jpg"),
    os.path.join(BACKGROUNDS_DIR, "08_artisan_atelier.jpg")
]


def load_json_file(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
        return default


def save_json_file(path: str, data: Any):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def get_lifestyle_path(item: Dict[str, Any]) -> str:
    return item.get("lifestyle_4k") or item.get("lifestyle_custom") or item.get("lifestyle") or ""


def sync_approved_folder():
    """Ensure all images marked 'approved' exist in saadaa_approved_lifestyle/."""
    reviews = load_json_file(REVIEW_STATUS_PATH, {})
    for path, meta in reviews.items():
        if meta.get("status") == "approved" and os.path.exists(path):
            fname = os.path.basename(path)
            dest = os.path.join(APPROVED_DIR, fname)
            if not os.path.exists(dest):
                try:
                    shutil.copy2(path, dest)
                except Exception as e:
                    logger.error(f"Failed to copy approved file {path}: {e}")


# Initialize approved folder sync
sync_approved_folder()


class VoteRequest(BaseModel):
    id: int
    action: str  # "like", "dislike", "reset"
    note: Optional[str] = None


class BatchVoteRequest(BaseModel):
    ids: List[int]
    action: str  # "like", "dislike", "reset"
    note: Optional[str] = None


class RegenItemRequest(BaseModel):
    id: int
    plate_index: Optional[int] = None


@router.get("/data")
async def get_gallery_data():
    """Fetch complete catalog with current review and queue statuses."""
    catalog = load_json_file(CATALOG_PATH, [])
    reviews = load_json_file(REVIEW_STATUS_PATH, {})
    queue = load_json_file(REGEN_QUEUE_PATH, [])
    queued_ids = set(q.get("id") for q in queue if isinstance(q, dict))

    counts = {"total": len(catalog), "approved": 0, "disliked": 0, "pending": 0}
    sku_stats: Dict[str, Dict[str, Any]] = {}
    categories = set()
    genders = set()

    enriched_items = []
    for item in catalog:
        item_id = item.get("id")
        sku = item.get("sku", "")
        color = item.get("color", "")
        cat = item.get("category", "General")
        gender = item.get("gender", "Unisex")
        categories.add(cat)
        genders.add(gender)

        lifestyle_rel = get_lifestyle_path(item)
        ecom_rel = item.get("ecom", "")

        # Determine review status
        review_entry = reviews.get(lifestyle_rel)
        if review_entry:
            status = review_entry.get("status", "pending")
        elif item_id in queued_ids:
            status = "disliked"
        else:
            status = "pending"

        counts[status] = counts.get(status, 0) + 1

        # Track SKU stats
        if sku not in sku_stats:
            sku_stats[sku] = {"sku": sku, "count": 0, "approved": 0, "disliked": 0, "pending": 0}
        sku_stats[sku]["count"] += 1
        sku_stats[sku][status] = sku_stats[sku].get(status, 0) + 1

        # Check resolution if exists
        is_4k = bool(item.get("lifestyle_4k")) or "4K" in lifestyle_rel

        enriched_items.append({
            "id": item_id,
            "sku": sku,
            "color": color,
            "color_code": item.get("color_code", ""),
            "color_abbr": item.get("color_abbr", ""),
            "category": cat,
            "gender": gender,
            "ecom_url": f"/{ecom_rel}" if ecom_rel and not ecom_rel.startswith("/") else ecom_rel,
            "ecom_path": ecom_rel,
            "lifestyle_url": f"/{lifestyle_rel}" if lifestyle_rel and not lifestyle_rel.startswith("/") else lifestyle_rel,
            "lifestyle_path": lifestyle_rel,
            "status": status,
            "is_4k": is_4k,
            "bg_name": item.get("bg_name", "Architectural Background"),
            "bg_short": item.get("bg_short", ""),
            "bg_tag": item.get("bg_tag", ""),
            "review_meta": review_entry
        })

    sorted_skus = sorted(list(sku_stats.values()), key=lambda x: x["sku"])

    return JSONResponse({
        "status": "success",
        "counts": counts,
        "total_skus": len(sorted_skus),
        "skus": sorted_skus,
        "categories": sorted(list(categories)),
        "genders": sorted(list(genders)),
        "items": enriched_items
    })


@router.post("/vote")
async def vote_item(req: VoteRequest):
    """Handle like (approve & save to folder), dislike (queue for regen), or reset."""
    catalog = load_json_file(CATALOG_PATH, [])
    item = next((it for it in catalog if it.get("id") == req.id), None)
    if not item:
        raise HTTPException(status_code=404, detail=f"Catalog item ID {req.id} not found")

    lifestyle_rel = get_lifestyle_path(item)
    filename = os.path.basename(lifestyle_rel)
    approved_file_path = os.path.join(APPROVED_DIR, filename)

    reviews = load_json_file(REVIEW_STATUS_PATH, {})
    queue = load_json_file(REGEN_QUEUE_PATH, [])
    # Clean queue to unique items by ID
    queue = [q for q in queue if isinstance(q, dict) and q.get("id") != req.id]

    action = req.action.lower()
    now = time.time()

    if action == "like":
        # Mark approved
        reviews[lifestyle_rel] = {
            "status": "approved",
            "timestamp": now,
            "sku": item.get("sku"),
            "color": item.get("color"),
            "note": req.note or "Approved in Gallery Review"
        }
        # Copy to approved directory
        if os.path.exists(lifestyle_rel):
            shutil.copy2(lifestyle_rel, approved_file_path)
            logger.info(f"Saved approved image to: {approved_file_path}")
        new_status = "approved"

    elif action == "dislike":
        # Mark disliked & send to regeneration queue
        reviews[lifestyle_rel] = {
            "status": "disliked",
            "timestamp": now,
            "sku": item.get("sku"),
            "color": item.get("color"),
            "note": req.note or "Disliked - Queued for regeneration"
        }
        # Add to regeneration queue
        queue.append({
            "id": req.id,
            "sku": item.get("sku"),
            "color": item.get("color"),
            "ecom": item.get("ecom"),
            "lifestyle_path": lifestyle_rel,
            "queued_at": now,
            "note": req.note or "User disliked"
        })
        # Remove from approved folder if present
        if os.path.exists(approved_file_path):
            try:
                os.remove(approved_file_path)
            except Exception as e:
                logger.warning(f"Error removing {approved_file_path}: {e}")
        new_status = "disliked"

    elif action == "reset":
        if lifestyle_rel in reviews:
            del reviews[lifestyle_rel]
        if os.path.exists(approved_file_path):
            try:
                os.remove(approved_file_path)
            except Exception as e:
                logger.warning(f"Error removing {approved_file_path}: {e}")
        new_status = "pending"

    else:
        raise HTTPException(status_code=400, detail=f"Invalid action '{action}'. Use like, dislike, or reset.")

    save_json_file(REVIEW_STATUS_PATH, reviews)
    save_json_file(REGEN_QUEUE_PATH, queue)

    # Re-calculate counts
    approved_count = sum(1 for v in reviews.values() if v.get("status") == "approved")
    disliked_count = sum(1 for v in reviews.values() if v.get("status") == "disliked")
    pending_count = len(catalog) - approved_count - disliked_count

    return {
        "status": "success",
        "item_id": req.id,
        "new_status": new_status,
        "counts": {
            "total": len(catalog),
            "approved": approved_count,
            "disliked": disliked_count,
            "pending": pending_count
        }
    }


@router.post("/batch_vote")
async def batch_vote(req: BatchVoteRequest):
    """Batch approve, dislike, or reset multiple items."""
    catalog = load_json_file(CATALOG_PATH, [])
    id_map = {it.get("id"): it for it in catalog}

    reviews = load_json_file(REVIEW_STATUS_PATH, {})
    queue = load_json_file(REGEN_QUEUE_PATH, [])
    queue_map = {q.get("id"): q for q in queue if isinstance(q, dict)}

    action = req.action.lower()
    now = time.time()
    processed_count = 0

    for item_id in req.ids:
        item = id_map.get(item_id)
        if not item:
            continue

        lifestyle_rel = get_lifestyle_path(item)
        filename = os.path.basename(lifestyle_rel)
        approved_file_path = os.path.join(APPROVED_DIR, filename)

        if action == "like":
            reviews[lifestyle_rel] = {
                "status": "approved",
                "timestamp": now,
                "sku": item.get("sku"),
                "color": item.get("color"),
                "note": req.note or "Batch approved"
            }
            if item_id in queue_map:
                del queue_map[item_id]
            if os.path.exists(lifestyle_rel):
                shutil.copy2(lifestyle_rel, approved_file_path)

        elif action == "dislike":
            reviews[lifestyle_rel] = {
                "status": "disliked",
                "timestamp": now,
                "sku": item.get("sku"),
                "color": item.get("color"),
                "note": req.note or "Batch disliked - queued for regen"
            }
            queue_map[item_id] = {
                "id": item_id,
                "sku": item.get("sku"),
                "color": item.get("color"),
                "ecom": item.get("ecom"),
                "lifestyle_path": lifestyle_rel,
                "queued_at": now,
                "note": req.note or "Batch disliked"
            }
            if os.path.exists(approved_file_path):
                try:
                    os.remove(approved_file_path)
                except Exception:
                    pass

        elif action == "reset":
            if lifestyle_rel in reviews:
                del reviews[lifestyle_rel]
            if item_id in queue_map:
                del queue_map[item_id]
            if os.path.exists(approved_file_path):
                try:
                    os.remove(approved_file_path)
                except Exception:
                    pass

        processed_count += 1

    save_json_file(REVIEW_STATUS_PATH, reviews)
    save_json_file(REGEN_QUEUE_PATH, list(queue_map.values()))

    approved_count = sum(1 for v in reviews.values() if v.get("status") == "approved")
    disliked_count = sum(1 for v in reviews.values() if v.get("status") == "disliked")
    pending_count = len(catalog) - approved_count - disliked_count

    return {
        "status": "success",
        "processed_count": processed_count,
        "counts": {
            "total": len(catalog),
            "approved": approved_count,
            "disliked": disliked_count,
            "pending": pending_count
        }
    }


@router.post("/regenerate_item")
async def regenerate_single_item(req: RegenItemRequest):
    """Regenerate a single item with an alternate architectural lifestyle background plate."""
    from scripts.process_all_drive_lifestyle import composite_garment_to_lifestyle

    catalog = load_json_file(CATALOG_PATH, [])
    item = next((it for it in catalog if it.get("id") == req.id), None)
    if not item:
        raise HTTPException(status_code=404, detail=f"Catalog item ID {req.id} not found")

    ecom_path = item.get("ecom", "")
    if not ecom_path or not os.path.exists(ecom_path):
        raise HTTPException(status_code=400, detail=f"Garment studio photo not found: {ecom_path}")

    lifestyle_path = get_lifestyle_path(item)
    if not lifestyle_path:
        lifestyle_path = os.path.join(
            PROJECT_ROOT,
            "saadaa_lifestyle_delivery",
            item.get("sku", "misc"),
            f"{item.get('sku')}_{item.get('color_code', 'GEN')}_lifestyle.jpg"
        )

    # Determine plate
    if req.plate_index is not None and 0 <= req.plate_index < len(BACKGROUND_PLATES):
        plate = BACKGROUND_PLATES[req.plate_index]
    else:
        # Cycle through background plates based on timestamp
        plate_idx = int(time.time()) % len(BACKGROUND_PLATES)
        plate = BACKGROUND_PLATES[plate_idx]

    os.makedirs(os.path.dirname(lifestyle_path), exist_ok=True)
    success = composite_garment_to_lifestyle(ecom_path, plate, lifestyle_path)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to composite regenerated lifestyle image.")

    # Remove from queue
    queue = load_json_file(REGEN_QUEUE_PATH, [])
    queue = [q for q in queue if isinstance(q, dict) and q.get("id") != req.id]
    save_json_file(REGEN_QUEUE_PATH, queue)

    # Update review status to pending
    reviews = load_json_file(REVIEW_STATUS_PATH, {})
    reviews[lifestyle_path] = {
        "status": "pending",
        "timestamp": time.time(),
        "sku": item.get("sku"),
        "color": item.get("color"),
        "note": f"Regenerated using plate: {os.path.basename(plate)}"
    }
    save_json_file(REVIEW_STATUS_PATH, reviews)

    return {
        "status": "success",
        "item_id": req.id,
        "lifestyle_path": lifestyle_path,
        "lifestyle_url": f"/{lifestyle_path}?t={int(time.time())}",
        "plate_used": os.path.basename(plate)
    }


@router.post("/regenerate_disliked")
async def regenerate_all_disliked(background_tasks: BackgroundTasks):
    """Regenerate all items currently queued in the regeneration queue."""
    from scripts.process_all_drive_lifestyle import composite_garment_to_lifestyle

    queue = load_json_file(REGEN_QUEUE_PATH, [])
    if not queue:
        return {"status": "ok", "message": "Regeneration queue is empty", "regenerated_count": 0}

    regenerated = []
    failed = []
    reviews = load_json_file(REVIEW_STATUS_PATH, {})

    for idx, item_q in enumerate(queue):
        if not isinstance(item_q, dict):
            continue
        ecom = item_q.get("ecom")
        lifestyle = item_q.get("lifestyle_path")
        if not ecom or not os.path.exists(ecom):
            failed.append(item_q.get("id"))
            continue

        plate = BACKGROUND_PLATES[(idx + 1) % len(BACKGROUND_PLATES)]
        ok = composite_garment_to_lifestyle(ecom, plate, lifestyle)
        if ok:
            regenerated.append(item_q.get("id"))
            reviews[lifestyle] = {
                "status": "pending",
                "timestamp": time.time(),
                "sku": item_q.get("sku"),
                "color": item_q.get("color"),
                "note": f"Batch regenerated using plate {os.path.basename(plate)}"
            }
        else:
            failed.append(item_q.get("id"))

    # Clear queue of successfully regenerated items
    remaining_queue = [q for q in queue if isinstance(q, dict) and q.get("id") in failed]
    save_json_file(REGEN_QUEUE_PATH, remaining_queue)
    save_json_file(REVIEW_STATUS_PATH, reviews)

    return {
        "status": "success",
        "total_queued": len(queue),
        "regenerated_count": len(regenerated),
        "failed_count": len(failed),
        "regenerated_ids": regenerated
    }


@router.get("/download_approved_zip")
async def download_approved_zip():
    """Package and serve all approved lifestyle images as a downloadable ZIP file."""
    sync_approved_folder()

    files = [os.path.join(APPROVED_DIR, f) for f in os.listdir(APPROVED_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        raise HTTPException(status_code=404, detail="No approved images found yet. Like some looks first!")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in files:
            fname = os.path.basename(fpath)
            zf.write(fpath, arcname=fname)

        # Include summary manifesto
        reviews = load_json_file(REVIEW_STATUS_PATH, {})
        approved_manifest = {
            "title": "SAADAA Approved Lifestyle Selects",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "approved_count": len(files),
            "approved_files": [os.path.basename(f) for f in files]
        }
        zf.writestr("APPROVED_MANIFEST.json", json.dumps(approved_manifest, indent=2))

    zip_buffer.seek(0)
    filename = f"Saadaa_Approved_Lifestyle_Picks_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@router.get("/queue")
async def get_regeneration_queue():
    """Retrieve all items currently queued for regeneration."""
    queue = load_json_file(REGEN_QUEUE_PATH, [])
    return JSONResponse({"status": "success", "count": len(queue), "queue": queue})
