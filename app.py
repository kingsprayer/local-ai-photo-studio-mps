"""FastAPI backend server for SAADAA Local AI Product Photo Studio.

Exposes REST APIs, WebSocket live log/progress streaming, model management,
dynamic style guide reloading, batch queue execution, and marketplace compliance auditing.
"""

import os
import sys
import io
import time
import json
import uuid
import shutil
import zipfile
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    BackgroundTasks,
    Request,
    Response
)
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import yaml
from PIL import Image

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from catalog.channels import CHANNEL_FORMATS, get_channel
from catalog.models_manager import get_all_models_status
from catalog.queue import job_queue
from catalog.vision import extract_attributes, describe_background
from catalog.pipeline import process_catalog_job, load_style_guide
import instagram_router
import gallery_router

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("saadaa.studio")

app = FastAPI(title="SAADAA Local AI Product Photo Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(instagram_router.router)
app.include_router(gallery_router.router)

# Ensure required directories exist
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Mount media and static folders
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

BACKGROUND_PLATES_DIR = os.path.join(PROJECT_ROOT, "background_plates")
LIFESTYLE_SAMPLES_DIR = os.path.join(STATIC_DIR, "lifestyle_samples")
MOODBOARDS_DIR = os.path.join(STATIC_DIR, "moodboards")
DRIVE_GARMENT_CACHE_DIR = os.path.join(PROJECT_ROOT, "drive_garment_cache")
SAADAA_ALL_DELIVERY_DIR = os.path.join(PROJECT_ROOT, "saadaa_all_lifestyle_delivery")
SAADAA_LIFESTYLE_DELIVERY_DIR = os.path.join(PROJECT_ROOT, "saadaa_lifestyle_delivery")
SAADAA_LIFESTYLE_IMAGES_DIR = os.path.join(PROJECT_ROOT, "saadaa_lifestyle_images")

if os.path.exists(BACKGROUND_PLATES_DIR):
    app.mount("/background_plates", StaticFiles(directory=BACKGROUND_PLATES_DIR), name="background_plates")
if os.path.exists(LIFESTYLE_SAMPLES_DIR):
    app.mount("/lifestyle_samples", StaticFiles(directory=LIFESTYLE_SAMPLES_DIR), name="lifestyle_samples")
if os.path.exists(MOODBOARDS_DIR):
    app.mount("/moodboards", StaticFiles(directory=MOODBOARDS_DIR), name="moodboards")
if os.path.exists(DRIVE_GARMENT_CACHE_DIR):
    app.mount("/drive_garment_cache", StaticFiles(directory=DRIVE_GARMENT_CACHE_DIR), name="drive_garment_cache")
if os.path.exists(SAADAA_ALL_DELIVERY_DIR):
    app.mount("/saadaa_all_lifestyle_delivery", StaticFiles(directory=SAADAA_ALL_DELIVERY_DIR), name="saadaa_all_lifestyle_delivery")
if os.path.exists(SAADAA_LIFESTYLE_DELIVERY_DIR):
    app.mount("/saadaa_lifestyle_delivery", StaticFiles(directory=SAADAA_LIFESTYLE_DELIVERY_DIR), name="saadaa_lifestyle_delivery")
if os.path.exists(SAADAA_LIFESTYLE_IMAGES_DIR):
    app.mount("/saadaa_lifestyle_images", StaticFiles(directory=SAADAA_LIFESTYLE_IMAGES_DIR), name="saadaa_lifestyle_images")

SAADAA_APPROVED_DIR = os.path.join(PROJECT_ROOT, "saadaa_approved_lifestyle")
os.makedirs(SAADAA_APPROVED_DIR, exist_ok=True)
app.mount("/saadaa_approved_lifestyle", StaticFiles(directory=SAADAA_APPROVED_DIR), name="saadaa_approved_lifestyle")

LIFESTYLE_INCOMING_DIR = os.path.join(PROJECT_ROOT, "lifestyle_incoming")
os.makedirs(LIFESTYLE_INCOMING_DIR, exist_ok=True)
app.mount("/lifestyle_incoming", StaticFiles(directory=LIFESTYLE_INCOMING_DIR), name="lifestyle_incoming")

# Connected WebSocket clients
active_websockets: List[WebSocket] = []


async def broadcast_ws(event_type: str, data: Any):
    """Broadcast JSON message to all connected WebSocket clients."""
    payload = json.dumps({"type": event_type, "data": data})
    disconnected = []
    for ws in active_websockets:
        try:
            await ws.send_text(payload)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in active_websockets:
            active_websockets.remove(ws)


def sync_job_listener(job: Dict[str, Any]):
    """Sync listener called from queue to queue broadcast in asyncio loop."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(broadcast_ws("job_update", job), loop)
    except Exception:
        pass


job_queue.add_listener(sync_job_listener)


# Background worker for processing jobs sequentially
async def worker_loop():
    """Continuously pull queued jobs and execute them through catalog.pipeline."""
    logger.info("SAADAA background generation worker started.")
    while True:
        try:
            job_id = await job_queue.queue.get()
            job = job_queue.get_job(job_id)
            if not job or job.get("status") == "cancelled":
                job_queue.queue.task_done()
                continue

            job_queue.update_job(
                job_id,
                status="processing",
                progress=5,
                step="Starting generation pipeline...",
                log_message=f"Starting job: {job['name']} (Mode: {job['mode']})"
            )

            def progress_callback(pct: int, step_name: str, message: str):
                job_queue.update_job(
                    job_id,
                    progress=pct,
                    step=step_name,
                    log_message=f"{step_name}: {message}"
                )

            try:
                # Load primary source image
                img_path = job["image_path"]
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Source image not found: {img_path}")
                source_image = Image.open(img_path).convert("RGB")

                # Handle background image if supplied
                options = dict(job.get("options", {}))
                bg_path = options.get("background_image_path")
                if bg_path and os.path.exists(bg_path):
                    options["background_image"] = Image.open(bg_path).convert("RGB")

                # Check if mock diffusion is explicitly enabled
                mock_mode = (
                    os.environ.get("SAADAA_MOCK_DIFFUSION") == "1" or
                    options.get("mock_mode") is True
                )

                # Process job through catalog pipeline
                result = process_catalog_job(
                    image_input=source_image,
                    job_name=job["name"],
                    mode=job["mode"],
                    channels=job["channels"],
                    attributes=job.get("attributes", {}),
                    options=options,
                    progress_callback=progress_callback,
                    mock_mode=mock_mode
                )

                job_queue.update_job(
                    job_id,
                    status="completed",
                    progress=100,
                    step="Catalog generation complete",
                    log_message=f"Successfully generated {len(result['channels'])} channel exports.",
                    results=result
                )

            except Exception as e:
                logger.exception(f"Error processing job {job_id}: {e}")
                retries = job.get("retries", 0)
                if retries < 2 and "CUDA" not in str(e):
                    job["retries"] = retries + 1
                    job_queue.update_job(
                        job_id,
                        status="queued",
                        step="Retrying job after failure...",
                        log_message=f"Failed with error: {str(e)}. Retrying ({job['retries']}/2)..."
                    )
                    job_queue.queue.put_nowait(job_id)
                else:
                    job_queue.update_job(
                        job_id,
                        status="failed",
                        step="Generation failed",
                        log_message=f"Fatal error: {str(e)}",
                        error=str(e)
                    )

            job_queue.queue.task_done()

        except Exception as e:
            logger.error(f"Worker exception: {e}")
            await asyncio.sleep(1)


@app.on_event("startup")
async def startup_event():
    import asyncio
    asyncio.create_task(worker_loop())
    asyncio.create_task(lifestyle_watcher_background_task())


async def lifestyle_watcher_background_task():
    """Background task continuously monitoring for new lifestyle images and broadcasting real-time updates."""
    logger.info("SAADAA continuous lifestyle image watcher task launched.")
    from scripts.lifestyle_watcher import lifestyle_watcher
    import asyncio
    while True:
        try:
            loop = asyncio.get_event_loop()
            updated_items, stats = await loop.run_in_executor(None, lifestyle_watcher.scan_once)
            if updated_items:
                logger.info(f"Lifestyle Watcher detected {len(updated_items)} updates. Broadcasting to {len(active_websockets)} clients...")
                await broadcast_ws("lifestyle_updated", {
                    "updated_items": updated_items,
                    "stats": stats,
                    "count": len(updated_items),
                    "timestamp": time.time()
                })
        except Exception as e:
            logger.error(f"Lifestyle watcher task error: {e}")
        await asyncio.sleep(2.5)


# Lifestyle Review & Approval Gallery (Default Landing Page)
@app.get("/")
@app.get("/gallery")
@app.get("/review")
async def serve_gallery():
    gallery_path = os.path.join(STATIC_DIR, "lifestyle_gallery.html")
    if not os.path.exists(gallery_path):
        gallery_path = os.path.join(PROJECT_ROOT, "lifestyle_gallery.html")
    if not os.path.exists(gallery_path):
        return JSONResponse({"status": "starting", "message": "Lifestyle Gallery loading..."})
    with open(gallery_path, "r", encoding="utf-8") as f:
        content = f.read()
    response = Response(content=content, media_type="text/html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/catalog-studio")
@app.get("/studio-old")
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return JSONResponse({"status": "starting", "message": "Index page loading..."})
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()
    response = Response(content=content, media_type="text/html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# Instagram Automation Suite Interface
@app.get("/instagram")
async def serve_instagram():
    ig_path = os.path.join(STATIC_DIR, "instagram.html")
    if not os.path.exists(ig_path):
        return JSONResponse({"status": "starting", "message": "Instagram Studio loading..."})
    with open(ig_path, "r", encoding="utf-8") as f:
        content = f.read()
    response = Response(content=content, media_type="text/html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# 4K Commercial Lifestyle AI Photography Studio (SOP v2.0 & 9 Fabric Moodboards)
@app.get("/lifestyle-studio")
@app.get("/lifestyle_studio")
@app.get("/lifestyle")
async def serve_lifestyle_studio():
    lifestyle_path = os.path.join(PROJECT_ROOT, "lifestyle_photography_studio.html")
    if not os.path.exists(lifestyle_path):
        lifestyle_path = os.path.join(STATIC_DIR, "lifestyle_studio.html")
    if not os.path.exists(lifestyle_path):
        return JSONResponse({"status": "starting", "message": "Lifestyle Studio loading..."})
    with open(lifestyle_path, "r", encoding="utf-8") as f:
        content = f.read()
    response = Response(content=content, media_type="text/html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# 3-Column Triptych Moodboard & Lookbook
@app.get("/moodboard")
@app.get("/moodboards")
@app.get("/triptych")
async def serve_moodboard():
    moodboard_path = os.path.join(PROJECT_ROOT, "moodboard.html")
    if not os.path.exists(moodboard_path):
        moodboard_path = os.path.join(STATIC_DIR, "moodboard.html")
    if not os.path.exists(moodboard_path):
        return JSONResponse({"status": "starting", "message": "Moodboard loading..."})
    with open(moodboard_path, "r", encoding="utf-8") as f:
        content = f.read()
    response = Response(content=content, media_type="text/html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# Moodboard Real-Time Dataset Endpoints
@app.get("/moodboard_data.js")
@app.get("/static/moodboard_data.js")
async def serve_moodboard_data_js():
    js_path = os.path.join(PROJECT_ROOT, "moodboard_data.js")
    if not os.path.exists(js_path):
        js_path = os.path.join(STATIC_DIR, "moodboard_data.js")
    with open(js_path, "r", encoding="utf-8") as f:
        content = f.read()
    response = Response(content=content, media_type="application/javascript")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@app.get("/moodboard_data.json")
@app.get("/static/moodboard_data.json")
async def serve_moodboard_data_json():
    json_path = os.path.join(PROJECT_ROOT, "moodboard_data.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(STATIC_DIR, "moodboard_data.json")
    with open(json_path, "r", encoding="utf-8") as f:
        content = f.read()
    response = Response(content=content, media_type="application/json")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@app.get("/Saadaa_Complete_All_Lifestyle_Images_Master.zip")
@app.get("/api/download/lifestyle_master_zip")
async def serve_unified_master_lifestyle_zip():
    zip_path = os.path.join(PROJECT_ROOT, "saadaa_lifestyle_delivery", "Saadaa_Complete_All_Lifestyle_Images_Master.zip")
    if not os.path.exists(zip_path):
        zip_path = os.path.join(PROJECT_ROOT, "Saadaa_Complete_All_Lifestyle_Images_Master.zip")
    if os.path.exists(zip_path):
        return FileResponse(zip_path, media_type="application/zip", filename="Saadaa_Complete_All_Lifestyle_Images_Master.zip")
    return JSONResponse({"status": "error", "message": "Unified Master zip archive not found"}, 404)


@app.get("/saadaa_all_lifestyle_images.zip")
@app.get("/static/saadaa_all_lifestyle_images.zip")
async def serve_master_lifestyle_zip():
    zip_path = os.path.join(PROJECT_ROOT, "saadaa_all_lifestyle_images.zip")
    if os.path.exists(zip_path):
        return FileResponse(zip_path, media_type="application/zip", filename="saadaa_all_lifestyle_images.zip")
    return JSONResponse({"status": "error", "message": "Master zip archive not found"}, 404)


@app.get("/api/moodboard/status")
async def get_moodboard_status():
    from scripts.lifestyle_watcher import lifestyle_watcher
    stats = lifestyle_watcher.get_stats()
    stats["active_websockets"] = len(active_websockets)
    return JSONResponse(stats)


@app.get("/api/moodboard/data")
async def get_moodboard_data():
    json_path = os.path.join(PROJECT_ROOT, "moodboard_data.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(data)


@app.post("/api/moodboard/scan")
async def trigger_moodboard_scan():
    from scripts.lifestyle_watcher import lifestyle_watcher
    import asyncio
    loop = asyncio.get_event_loop()
    updated_items, stats = await loop.run_in_executor(None, lifestyle_watcher.scan_once)
    if updated_items:
        await broadcast_ws("lifestyle_updated", {
            "updated_items": updated_items,
            "stats": stats,
            "count": len(updated_items),
            "timestamp": time.time()
        })
    return {"status": "ok", "updated_count": len(updated_items), "updated_items": updated_items, "stats": stats}


@app.post("/api/moodboard/upload_lifestyle")
async def upload_lifestyle_images(files: List[UploadFile] = File(...)):
    """Upload new lifestyle images or zip file directly and trigger immediate sync."""
    incoming_dir = os.path.join(PROJECT_ROOT, "lifestyle_incoming")
    os.makedirs(incoming_dir, exist_ok=True)
    saved_files = []
    for upload in files:
        target_path = os.path.join(incoming_dir, upload.filename)
        with open(target_path, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved_files.append(target_path)
    
    from scripts.lifestyle_watcher import lifestyle_watcher
    import asyncio
    loop = asyncio.get_event_loop()
    updated_items, stats = await loop.run_in_executor(None, lifestyle_watcher.scan_once)
    if updated_items:
        await broadcast_ws("lifestyle_updated", {
            "updated_items": updated_items,
            "stats": stats,
            "count": len(updated_items),
            "timestamp": time.time()
        })
    return {"status": "success", "saved_files": len(saved_files), "updated_items": updated_items, "stats": stats}


# WebSocket endpoint for real-time live events and logging
@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        # Send initial status
        await websocket.send_text(json.dumps({
            "type": "init",
            "data": {
                "status": "connected",
                "models": get_all_models_status(),
                "pending_jobs": len(job_queue.memory_jobs)
            }
        }))
        while True:
            # Keep-alive ping/pong
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
    except Exception:
        if websocket in active_websockets:
            active_websockets.remove(websocket)


# System & models status
@app.get("/api/status")
async def get_system_status():
    models_status = get_all_models_status()
    
    # Check Ollama status
    import requests
    ollama_ok = False
    gemma_ok = False
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        if r.status_code == 200:
            ollama_ok = True
            models_list = [m.get("name", "") for m in r.json().get("models", [])]
            gemma_ok = any("gemma3:4b" in m for m in models_list)
    except Exception:
        pass

    import torch
    mps_available = torch.backends.mps.is_available()

    return {
        "system": {
            "platform": "Apple Silicon (MPS)",
            "mps_available": mps_available,
            "ollama_running": ollama_ok,
            "gemma3_available": gemma_ok,
            "mock_mode": os.environ.get("SAADAA_MOCK_DIFFUSION") == "1"
        },
        "models": models_status["models"],
        "features": models_status["features"],
        "all_commercial_licensed": models_status["all_commercial_licensed"],
        "queue": {
            "total": len(job_queue.memory_jobs),
            "queued": len([j for j in job_queue.memory_jobs.values() if j.get("status") == "queued"]),
            "processing": len([j for j in job_queue.memory_jobs.values() if j.get("status") == "processing"]),
            "completed": len([j for j in job_queue.memory_jobs.values() if j.get("status") == "completed"]),
            "failed": len([j for j in job_queue.memory_jobs.values() if j.get("status") == "failed"]),
        }
    }


# Channel specifications endpoint
@app.get("/api/channels")
async def list_channel_specs():
    return CHANNEL_FORMATS


# Style Guide endpoints
@app.get("/api/style-guide")
async def get_style_guide_endpoint():
    guide = load_style_guide()
    return guide


@app.post("/api/style-guide")
async def save_style_guide_endpoint(payload: Dict[str, Any]):
    guide_path = os.path.join(PROJECT_ROOT, "style_guide.yaml")
    try:
        with open(guide_path, "w", encoding="utf-8") as f:
            yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
        return {"status": "success", "message": "Brand style guide saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save style guide: {e}")


# Vision attribute extraction
@app.post("/api/extract-attributes")
async def extract_attributes_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        attrs = extract_attributes(img)
        return attrs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attribute extraction failed: {e}")


# Upload image (supports single, batch, or chunked upload)
@app.post("/api/upload")
async def upload_image_endpoint(
    file: UploadFile = File(...),
    folder: Optional[str] = Form("catalog"),
    batch_prefix: Optional[str] = Form(None)
):
    try:
        sanitized_folder = os.path.normpath(folder).lstrip("/").replace("..", "_")
        dest_dir = os.path.join(UPLOADS_DIR, sanitized_folder)
        os.makedirs(dest_dir, exist_ok=True)

        filename = file.filename or f"image_{int(time.time())}.jpg"
        if batch_prefix:
            filename = f"{batch_prefix}_{filename}"
        
        file_path = os.path.join(dest_dir, filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Get image dimensions
        with Image.open(file_path) as img:
            w, h = img.size

        rel_path = os.path.relpath(file_path, PROJECT_ROOT)
        return {
            "status": "success",
            "file_path": file_path,
            "relative_url": f"/{rel_path}",
            "filename": filename,
            "folder": sanitized_folder,
            "width": w,
            "height": h,
            "size_bytes": os.path.getsize(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


# Create new jobs or batch
@app.post("/api/jobs")
async def create_jobs_endpoint(payload: Dict[str, Any]):
    """Register one or more jobs into the persisted disk queue.
    
    Expected payload:
    - items: List of dicts, each with:
      - image_path: str
      - job_name: str (e.g. 'summer_kurta/sku01.jpg')
      - mode: 'packshot' | 'scene' | 'model'
      - channels: List[str]
      - attributes: Dict[str, Any]
      - options: Dict[str, Any]
    - batch_id: Optional[str]
    """
    items = payload.get("items", [])
    batch_id = payload.get("batch_id") or f"batch_{str(uuid.uuid4())[:8]}"
    created_jobs = []

    if not items:
        # Single job payload fallback
        single_job = {
            "image_path": payload.get("image_path"),
            "job_name": payload.get("job_name", "product.jpg"),
            "mode": payload.get("mode", "packshot"),
            "channels": payload.get("channels", ["amazon", "myntra", "shopify"]),
            "attributes": payload.get("attributes", {}),
            "options": payload.get("options", {})
        }
        items = [single_job]

    for it in items:
        img_path = it.get("image_path")
        if not img_path or not os.path.exists(img_path):
            continue

        job = job_queue.create_job(
            image_path=img_path,
            job_name=it.get("job_name", "product.jpg"),
            mode=it.get("mode", "packshot"),
            channels=it.get("channels", ["amazon", "myntra", "shopify"]),
            attributes=it.get("attributes", {}),
            options=it.get("options", {}),
            batch_id=batch_id
        )
        created_jobs.append(job)

    return {
        "status": "success",
        "batch_id": batch_id,
        "count": len(created_jobs),
        "jobs": created_jobs
    }


# Query jobs
@app.get("/api/jobs")
async def list_jobs_endpoint(batch_id: Optional[str] = None):
    return job_queue.list_jobs(batch_id=batch_id)


@app.get("/api/jobs/{job_id}")
async def get_job_endpoint(job_id: str):
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/jobs/{job_id}/retry")
async def retry_job_endpoint(job_id: str):
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job["status"] = "queued"
    job["progress"] = 0
    job["current_step"] = "Retrying in queue..."
    job["log"].append(f"[{time.strftime('%H:%M:%S')}] User requested manual retry.")
    job_queue.save_job_to_disk(job)
    job_queue.queue.put_nowait(job_id)
    return {"status": "success", "message": f"Job {job_id} re-enqueued."}


# Output Library Explorer
@app.get("/api/library")
async def get_library_endpoint(
    folder: Optional[str] = None,
    mode: Optional[str] = None,
    query: Optional[str] = None
):
    """Scan outputs/ directory and return all catalog products with meta.json."""
    items = []
    if not os.path.exists(OUTPUTS_DIR):
        return {"items": []}

    for root, _, files in os.walk(OUTPUTS_DIR):
        if "meta.json" in files:
            meta_path = os.path.join(root, "meta.json")
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                
                # Check filters
                if folder and meta.get("folder") != folder:
                    continue
                if mode and meta.get("mode") != mode:
                    continue
                if query:
                    q = query.lower()
                    title = meta.get("attributes", {}).get("seo_title", "").lower()
                    sku = meta.get("sku", "").lower()
                    cat = meta.get("attributes", {}).get("category", "").lower()
                    if q not in title and q not in sku and q not in cat:
                        continue

                items.append(meta)
            except Exception as e:
                logger.error(f"Error parsing meta {meta_path}: {e}")

    # Sort newest first
    items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return {"items": items, "count": len(items)}


# Download ZIP archive of an entire catalog export
@app.get("/api/download-zip/{folder}/{sku}/{mode}")
async def download_zip_endpoint(folder: str, sku: str, mode: str):
    target_dir = os.path.join(OUTPUTS_DIR, folder, sku, mode)
    if not os.path.exists(target_dir):
        raise HTTPException(status_code=404, detail="Product output directory not found")

    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(target_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, target_dir)
                zf.write(full_path, rel_path)

    mem_zip.seek(0)
    zip_filename = f"SAADAA_{sku}_{mode}_catalog.zip"
    return StreamingResponse(
        mem_zip,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'}
    )


# Creator Benchmark Analysis & Style Strategy Endpoint
@app.get("/api/creator-benchmark")
async def get_creator_benchmark_endpoint():
    """Returns strategic creator benchmark analysis for SAADAA vs peer fashion brands."""
    return {
        "brand": "SAADAA",
        "market": "Indian Premium Contemporary Apparel & Minimalist Everyday Luxury",
        "benchmark_targets": ["Nicobar", "FabIndia", "Zara", "Westside", "Anokhi"],
        "viral_rate_and_capacity": {
            "optimal_content_cadence": "4-6 catalog looks/day, batch scheduled",
            "golden_hours_ist": ["08:30 Morning", "13:00 Lunch", "20:30 Prime Evening"],
            "engagement_rate_target": "5.4% on Reels & Myntra Live Storefronts"
        },
        "topic_quadrants": {
            "high_engagement_low_barrier": "Fabric breathability tests, linen close-up macro weave, fit comparisons on diverse body types",
            "high_shares_strong_point_of_view": "Sustainable natural Indian dyes, authentic slow fashion craftsmanship, capsule wardrobe tutorials",
            "high_conversion_commercial": "Full marketplace catalog multi-angle packshots, smart cropped Myntra & Amazon storefront sets"
        },
        "visual_style_dna": {
            "lighting": "Diffused morning sunlight with architectural shadow lines (zero harsh direct strobe)",
            "mannequin_to_model": "Natural human poses with 18-point OpenPose conditioning, maintaining 100% garment weave fidelity",
            "color_signature": "Warm Ivory, Terracotta, Desert Sand, Indigo Slub"
        }
    }


# ==============================================================================
# 4K LIFESTYLE AI PHOTOGRAPHY (SOP v2.0) API ENDPOINTS
# ==============================================================================

LIFESTYLE_MOODBOARDS = {
    "MB01": {
        "id": "MB01",
        "code": "MB-01",
        "title": "Tiered Maxi / Bohemian Resort Living",
        "slot_alias": "Resort Editorial (Tiered Dresses)",
        "fabric": "Flowy Mulmul, Lightweight Linen & Cotton Voile",
        "category": "resort",
        "kelvin": "3600K Golden Sunlit",
        "assigned_rig": "Hasselblad X2D II 100C",
        "plate_image": "/static/moodboards/mb01_resort_tiered_maxi.jpg",
        "vibe": "Mediterranean villa limestone terrace, ancient olive trees casting dappled shadows, rustic terracotta amphora urns, whitewashed arches, distant sea horizon, warm late-afternoon natural sunlight (3600K).",
        "posture": "Breezy walking stride or seated comfortably on limestone steps with relaxed natural posture, kinetic fabric billow catching light, hand gently touching fabric tiers.",
        "optics_signature": "HNCS 16-bit color fidelity, organic highlight rolloff across airy tiered ruffles, authentic medium-format 3D pop without digital oversharpening.",
        "setting_elements": ["Mediterranean limestone terrace", "Olive trees & dappled shade", "Terracotta amphora urns", "Whitewashed arches", "Sea view horizon", "Clean stone steps"],
        "colorways": ["Ivory White", "Wine Burgundy", "Soft Lavender", "Jet Black", "Dusty Rose", "Olive Green", "Soft Butter Yellow", "Sky Blue"]
    },
    "MB02": {
        "id": "MB02",
        "code": "MB-02",
        "title": "T-Shirt Collection / Casual & Sporty",
        "slot_alias": "Casual & Sporty (Active T-Shirts)",
        "fabric": "100% Combed Cotton, Cotton Pique & Jersey Knits",
        "category": "casual-active",
        "kelvin": "5500K Clear Morning Daylight",
        "assigned_rig": "Sony Alpha 7R V",
        "plate_image": "/static/moodboards/mb02_casual_sporty_tshirts.jpg",
        "vibe": "Open-air landscaped city park, lush green lawns, tennis and basketball court perimeter, clean concrete jogging tracks, minimal architectural walls and benches, clear morning sunlight (5500K).",
        "posture": "Athletic dynamic lifestyle moment: mid-stride with sports tote bag, sitting on park bench sipping from water bottle, relaxed cross-legged on lawn, stretching or holding basketball/racket.",
        "optics_signature": "BIONZ XR 61MP high micro-contrast resolving breathable cotton knit ribbing, crisp crew neckline acutance, authentic athletic skin micro-pores.",
        "setting_elements": ["Parks & green lawns", "Tennis & basketball courts", "Jogging tracks", "Outdoor wooden benches", "Steps & ledges", "Minimal architectural walls", "Water bottles & canvas totes", "Caps & clean sneakers"],
        "colorways": ["Dusty Rose / Mauve", "Olive Green", "Crisp White", "Slate Blue", "Classic Black", "Buttercup Yellow", "Deep Burgundy", "Soft Lilac"]
    },
    "MB03": {
        "id": "MB03",
        "code": "MB-03",
        "title": "Viscose / Soft & Fluid Urban Lifestyle",
        "slot_alias": "Urban Lifestyle (Fluid Co-Ords & Tunics)",
        "fabric": "Fluid Viscose, Modal, Rayon Crepe & Silk Blends",
        "category": "urban",
        "kelvin": "5200K Soft Diffused Skylight",
        "assigned_rig": "Sony Alpha 7R V",
        "plate_image": "/static/moodboards/mb03_viscose_urban_lifestyle.jpg",
        "vibe": "Modern city spaces, boutique sidewalk cafes and coffee corners, textured stone pillars, large glass doors and window reflections, indoor/outdoor potted plants, clean Scandinavian furniture, soft diffused daylight (5200K).",
        "posture": "Effortless urban stride with leather tote, relaxed sitting at outdoor cafe table enjoying coffee, reading a book on a wooden bench, natural fluid drape cascading across silhouette.",
        "optics_signature": "Continuous smooth tonal gradations over fluid viscose drapery, crisp shadow detail without clipping, high acutance on collar and welt seams.",
        "setting_elements": ["Modern city spaces", "Cafes & coffee corners", "Minimal architecture", "Textured walls & pillars", "Glass doors & windows", "Indoor plants", "Clean modern furniture", "Open airy spaces"],
        "colorways": ["Ecru Cream", "Deep Burgundy / Wine", "Soft Lilac", "Olive Green", "Dusty Pink", "Jet Black", "Pastel Yellow", "Crisp White", "Periwinkle Blue", "Navy"]
    },
    "MB04": {
        "id": "MB04",
        "code": "MB-04",
        "title": "Linen Blend / Everyday Courtyard Living",
        "slot_alias": "Courtyard Living (Everyday Relaxed Sets)",
        "fabric": "Linen-Cotton Blend, Slub Linen & Breathable Hemp",
        "category": "courtyard",
        "kelvin": "4200K Warm Afternoon",
        "assigned_rig": "Hasselblad X2D II 100C",
        "plate_image": "/static/moodboards/mb04_linen_blend_courtyard.jpg",
        "vibe": "Rustic courtyard with textured plaster walls and organic tree shadows, antique wooden doors and benches, large terracotta planter pots, natural stone flagstone flooring, warm afternoon breeze (4200K).",
        "posture": "Relaxed lived-in courtyard postures: seated on low stone steps, leaning casually against wooden doorframe holding straw bag, candid turn with relaxed tension creases at elbows.",
        "optics_signature": "Tactile European linen slub cross-weave, organic relaxed natural wrinkles, progressive optical bokeh falloff with warm medium-format dimensionality.",
        "setting_elements": ["Courtyard walls & shadows", "Terracotta pots", "Wooden doors & benches", "Minimal architecture", "Natural stone flooring", "Potted greenery", "Woven straw baskets", "Neutral & earthy textures"],
        "colorways": ["Natural Ivory", "Dusty Rose", "Navy Blue", "Olive Green", "Soft Lavender", "Jet Black", "Warm Ochre", "Crisp White"]
    },
    "MB05": {
        "id": "MB05",
        "code": "MB-05",
        "title": "Airy Linen / Minimal Outdoor Architecture",
        "slot_alias": "Minimal Editorial (Linen Tunics & Sets)",
        "fabric": "Pure Airy Linen, Crisp Flax & Handloom Muslin",
        "category": "minimal-architectural",
        "kelvin": "4800K Neutral Directional",
        "assigned_rig": "Sony Alpha 7R V",
        "plate_image": "/static/moodboards/mb05_airy_linen_minimal_architecture.jpg",
        "vibe": "Monolithic beige and warm grey micro-cement walls, clean geometric architecture, sculptural open doorways and round arches, smooth stone flooring, solitary terracotta amphora, subtle uncluttered backdrop, soft directional sunlight (4800K).",
        "posture": "Poised architectural stance, contemplative gaze, seated on geometric steps with posture highlighting the tunic notch neckline, clean side slits, and flowing white trousers.",
        "optics_signature": "Razor-sharp acutance on linen weave and notch collar structure, high dynamic range preserving deep shadow details against bright limestone.",
        "setting_elements": ["Beige & grey walls", "Minimal architecture", "Clean geometric lines", "Stone flooring", "Terracotta pots (minimal)", "Arches & open doorways", "Plain textured walls", "Open terraces", "Subtle clean backdrops"],
        "colorways": ["Off-White Ecru", "Rich Wine / Maroon", "Soft Lilac", "Olive Green", "Sky Blue", "Dusty Rose", "Warm Ochre", "Jet Black"]
    },
    "MB06": {
        "id": "MB06",
        "code": "MB-06",
        "title": "100% Cotton / Relaxed Garden Outdoors",
        "slot_alias": "Garden Outdoors (Everyday Cotton Tees)",
        "fabric": "100% Pure Organic Cotton & Breathable Slub Jersey",
        "category": "garden-casual",
        "kelvin": "5000K Diffused Outdoor",
        "assigned_rig": "Sony Alpha 7R V",
        "plate_image": "/static/moodboards/mb06_cotton_relaxed_garden.jpg",
        "vibe": "Lush garden lawn, shaded leafy tree canopies casting gentle dappled light, neutral rendered garden walls, weathered wooden benches, limestone steps, calm airy outdoor atmosphere (5000K).",
        "posture": "Relaxed outdoor ease: sitting cross-legged on green grass, leaning back on stone steps with book in hand, casual walk with woven market tote, unposed authentic smile.",
        "optics_signature": "Authentic matte finish of unmercerized organic cotton, true-to-life skin micro-pores under natural garden illumination.",
        "setting_elements": ["Garden lawn", "Minimal architecture", "Neutral walls", "Stone steps & ledges", "Wooden benches", "Trees & lush greenery", "Open outdoors", "Soft natural light", "Clean airy backgrounds"],
        "colorways": ["Crisp White", "Blush Pink", "Sky Blue", "Olive Drab", "Deep Maroon", "Jet Black", "Buttercup Yellow", "Soft Lilac"]
    },
    "MB07": {
        "id": "MB07",
        "code": "MB-07",
        "title": "Office / Formal Wear",
        "slot_alias": "Modern Corporate (Formal Shirts & Trousers)",
        "fabric": "Crisp Oxford Cotton, Tailored Poplin & Stretch Twill",
        "category": "formal-office",
        "kelvin": "5400K Clean Corporate Daylight",
        "assigned_rig": "Hasselblad X2D II 100C",
        "plate_image": "/static/moodboards/mb07_office_formal_wear.jpg",
        "vibe": "Sleek contemporary corporate architecture, floor-to-ceiling glass partitions, warm walnut and oak executive conference tables, clean desks with chrome accents, architectural shadow play from Venetian blinds, soft diffused daylight (5400K).",
        "posture": "Poised modern executive: walking through office corridor with slim laptop or leather portfolio, standing by window ledge with ceramic coffee mug, seated engaged at conference desk.",
        "optics_signature": "HNCS natural color science with 16-bit depth rendering immaculate white and pastel shirt collars, precise pleat sharpness on trousers, zero distortion on buttons.",
        "setting_elements": ["Modern office interiors", "Clean desks", "Minimal furniture", "Natural light", "Neutral walls", "Wood & stone textures", "Glass partitions", "Subtle indoor greenery", "Uncluttered spaces"],
        "colorways": ["Pristine White", "Espresso Dark Brown", "Ice Blue", "Camel / Taupe", "Jet Black", "Blush Pink", "Ivory Cream", "Charcoal Grey"]
    },
    "MB08": {
        "id": "MB08",
        "code": "MB-08",
        "title": "Pure Linen / Beach and Vacation Vibes",
        "slot_alias": "Coastal Vacation (Resort Tunics & Palazzos)",
        "fabric": "100% Pure European Flax Linen & Washed Gauze",
        "category": "beach-resort",
        "kelvin": "4500K Coastal Morning Sun",
        "assigned_rig": "Sony Alpha 7R V",
        "plate_image": "/static/moodboards/mb08_pure_linen_beach_vacation.jpg",
        "vibe": "Open Mediterranean / Goan coastline, sunlit sandy shores, sparkling ocean waves in background, whitewashed low perimeter walls, weathered wooden sun decks, wide open skies, crisp coastal morning sunlight (4500K).",
        "posture": "Sitting on coastal stone ledge enjoying sea breeze, walking along wooden boardwalk holding straw sun hat, relaxed holiday laughter, fabric fluttering naturally in ocean air.",
        "optics_signature": "High micro-contrast capturing authentic linen weave texture against sunny sea spray, progressive optical bokeh falloff with realistic contact shadows.",
        "setting_elements": ["Beaches & coastline", "Sandy shores", "Minimal architecture", "Whitewashed walls", "Wooden decks", "Stone steps & ledges", "Resort style spaces", "Open skies", "Woven bags & sun hats"],
        "colorways": ["Sand Ecru", "Deep Burgundy / Wine", "Soft Purple / Lilac", "Olive Green", "Bright Ocean Blue", "Pastel Rose", "Sunny Ochre Yellow", "Crisp White", "Classic Black"]
    }
}

# Backward compatibility aliases
LIFESTYLE_MOODBOARDS["FB01"] = LIFESTYLE_MOODBOARDS["MB01"]
LIFESTYLE_MOODBOARDS["FB02"] = LIFESTYLE_MOODBOARDS["MB04"]
LIFESTYLE_MOODBOARDS["FB03"] = LIFESTYLE_MOODBOARDS["MB06"]
LIFESTYLE_MOODBOARDS["FB04"] = LIFESTYLE_MOODBOARDS["MB02"]
LIFESTYLE_MOODBOARDS["FB05"] = LIFESTYLE_MOODBOARDS["MB05"]
LIFESTYLE_MOODBOARDS["FB06"] = LIFESTYLE_MOODBOARDS["MB03"]
LIFESTYLE_MOODBOARDS["FB07"] = LIFESTYLE_MOODBOARDS["MB07"]
LIFESTYLE_MOODBOARDS["FB08"] = LIFESTYLE_MOODBOARDS["MB08"]
LIFESTYLE_MOODBOARDS["FB09"] = LIFESTYLE_MOODBOARDS["MB02"]

DEMOGRAPHIC_PROFILES = {
    "Profile A": {
        "id": "Profile A",
        "name": "Pan-Indian Contemporary",
        "complexion": "Wheatish to warm honey skin tone, visible fine skin pores, subtle epidermal subsurface scattering",
        "grooming": {
            "men": "Faded short sides, textured matte styling, light groomed stubble",
            "women": "Soft beach waves, dewy minimalist makeup, visible pores"
        },
        "garment_affinity": ["Casual linen shirts", "Relaxed co-ords", "Modern loungewear"]
    },
    "Profile B": {
        "id": "Profile B",
        "name": "North / North-West Indian",
        "complexion": "Warm golden to light bronze tone, authentic skin warmth",
        "grooming": {
            "men": "Defined jawline, neat boxed beard, structured side part",
            "women": "Sleek low chignon, soft kajal eye accents, warm radiance"
        },
        "garment_affinity": ["Formal Bandhgalas", "Raw silk sherwanis", "Festive lehengas"]
    },
    "Profile C": {
        "id": "Profile C",
        "name": "South Indian / Dravidian",
        "complexion": "Rich deep bronze to warm olive skin tone, radiant natural glow",
        "grooming": {
            "men": "Clean-shaven or groomed classic mustache, radiant skin",
            "women": "Lustrous natural dark hair, refined traditional grace"
        },
        "garment_affinity": ["Kanjeevaram & Chanderi silks", "Occasion wear", "Kurtas"]
    },
    "Profile D": {
        "id": "Profile D",
        "name": "North-East Indian",
        "complexion": "Luminous ivory-honey complexion, glass-skin clarity with visible pores",
        "grooming": {
            "men": "Modern textured crop, clean-shaven, high cheekbones",
            "women": "Natural soft curtain bangs, high-fashion bone structure, expressive confident eyes"
        },
        "garment_affinity": ["Minimalist avant-garde", "Resort wear", "Luxury capsule items"]
    }
}

CAMERA_RIGS = {
    "sony": {
        "id": "sony",
        "name": "Sony Alpha 7R V",
        "badge": "Commercial Precision",
        "sensor": "61MP Full-Frame Exmor R BSI CMOS (35.7 x 23.8 mm)",
        "engine": "BIONZ XR image processing",
        "lenses": [
            "Sony FE 85mm f/1.4 GM II (f/2.0 - f/2.8) for 3/4 portraits and fashion editorial",
            "Sony FE 50mm f/1.2 GM (f/2.2) for full-body environmental lifestyle"
        ],
        "signature": "Tactile textile texture, razor-sharp eye autofocus, progressive spherical aberration bokeh falloff."
    },
    "hasselblad": {
        "id": "hasselblad",
        "name": "Hasselblad X2D II 100C",
        "badge": "Luxury Medium Format",
        "sensor": "100MP Medium Format (43.8 x 32.9 mm) BSI CMOS",
        "engine": "Hasselblad Natural Colour Solution (HNCS) with 16-bit color depth (281T colors) and 15 stops DR",
        "lenses": [
            "Hasselblad XCD 90mm f/2.5 V for three-quarter luxury fashion portraits",
            "Hasselblad XCD 55mm f/2.5 V for natural-perspective environmental compositions"
        ],
        "signature": "Medium-format '3D pop', organic highlight rolloff on silks and velvets, natural non-digital micro-relief on threads."
    }
}


@app.get("/api/lifestyle/moodboards")
async def get_lifestyle_moodboards():
    """Returns all 9 fabric-calibrated moodboards in the visual reference registry."""
    return {"moodboards": list(LIFESTYLE_MOODBOARDS.values()), "count": len(LIFESTYLE_MOODBOARDS)}


@app.get("/api/lifestyle/demographics")
async def get_lifestyle_demographics():
    """Returns the 4 Indian demographic casting archetypes."""
    return {"profiles": list(DEMOGRAPHIC_PROFILES.values())}


@app.get("/api/lifestyle/rigs")
async def get_lifestyle_rigs():
    """Returns flagship camera rig optics specifications."""
    return {"rigs": CAMERA_RIGS}


@app.post("/api/lifestyle/generate-prompt")
async def generate_lifestyle_prompt_endpoint(payload: Dict[str, Any]):
    """Synthesizes a master production prompt conforming to SOP v2.0 specifications."""
    mb_id = payload.get("moodboard_id", "MB01")
    mb = LIFESTYLE_MOODBOARDS.get(mb_id, LIFESTYLE_MOODBOARDS["MB01"])
    demo_id = payload.get("demographic_profile", "Profile B")
    demo = DEMOGRAPHIC_PROFILES.get(demo_id, DEMOGRAPHIC_PROFILES["Profile B"])
    rig_id = "hasselblad" if "Hasselblad" in payload.get("camera_rig", "") else "sony"
    rig = CAMERA_RIGS[rig_id]
    gender = payload.get("model_gender", "female")
    age = payload.get("model_age", 27)
    drive_link = payload.get("drive_link", "DRIVE_ROOT_04/Subfolder_SKU_W_LIN_042/01_garment_front_flatlay.jpg")
    garment_details = payload.get("garment_details", "exact pure linen relaxed tunic and wide trousers")
    colorway = payload.get("colorway", "")
    all_colorways = payload.get("all_colorways", mb.get("colorways", []))

    grooming_note = demo["grooming"]["men"] if gender == "male" else demo["grooming"]["women"]
    color_desc = f"{colorway} " if colorway else ""

    def build_single_prompt(col_name=""):
        c_str = f"{col_name} " if col_name else ""
        return (
            f"/imagine prompt: An ultra-luxurious 4K commercial fashion editorial photograph of a {age}-year-old "
            f"{demo['name']} {gender} model wearing the {c_str}{garment_details} from [Drive Link: {drive_link}]. "
            f"The garment's cut, neckline, buttons, drape, print scale, and fabric texture must be preserved with 100% precision. "
            f"Model Profile: {demo['complexion']}, styled with {grooming_note}. "
            f"Background Reference: Strictly replicating environment and lighting from Moodboard '{mb['title']}' ({mb['vibe']}). "
            f"The garment silhouette and color are sourced strictly from Drive e-com assets. "
            f"Interaction: {mb['posture']} "
            f"Camera & Optics: Shot on {rig['name']} with {rig['signature']} {mb['optics_signature']} "
            f"4K UHD, 3840 x 2160, 300 DPI commercial lookbook quality."
        )

    prompt_body = build_single_prompt(colorway)

    negative_constraints = (
        "--no cigarette, cigar, smoke, vape, lighter, tobacco, alcohol, wine, beer, cocktail, bar, "
        "deformed fingers, extra limbs, plastic skin, airbrush, CGI render, cartoon, altered garment cut, "
        "changed buttons, modified fabric pattern, lowres, blur --ar 4:5 --v 6.1 --style raw"
    )

    batch_prompts = []
    if all_colorways:
        for c in all_colorways:
            p_body = build_single_prompt(c)
            batch_prompts.append({
                "colorway": c,
                "prompt": f"{p_body} {negative_constraints}",
                "prompt_body": p_body
            })

    return {
        "status": "success",
        "master_prompt": f"{prompt_body} {negative_constraints}",
        "prompt_body": prompt_body,
        "negative_constraints": negative_constraints,
        "moodboard": mb,
        "moodboard_id": mb_id,
        "moodboard_title": mb.get("title", ""),
        "moodboard_code": mb.get("code", ""),
        "demographic": demo,
        "rig": rig,
        "colorway": colorway,
        "batch_colorways_prompts": batch_prompts
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
