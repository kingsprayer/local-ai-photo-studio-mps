# Local AI Photo Studio (Apple Silicon MPS)

Self-hosted e-commerce & lifestyle **4K AI Photography Studio** engineered for **Apple Silicon (PyTorch MPS)** — zero per-image cloud API costs. Supports Packshot (pure white RGB 255 + contact shadows), Scene, and Model generation modes with live WebSocket job streaming, Three.js OpenPose editing, and multi-marketplace compliance QA.

## Key Capabilities
- **Modular Diffusion & CV Pipeline (`catalog/`):** End-to-end orchestration across garment detection (`detect.py`), segmentation & alpha matting (`masks.py`), 18-point OpenPose conditioning (`pose.py`), SDXL / RealVisXL V4.0 + ControlNet generation (`generator.py`), 4x upscaling (`upscale.py`), compositing (`compose.py`), and 7-marketplace compliance checking (`compliance.py`, `channels.py`).
- **FastAPI + WebSocket Studio (`app.py`, `static/index.html`, `static/pose-studio.js`):** Interactive browser studio with Three.js 3D OpenPose rig, batch queue (`batch.py`, `catalog/queue.py`), and gallery review router (`gallery_router.py`).
- **Local Vision QA (`catalog/vision.py`):** Supports local Ollama vision models (`gemma3:4b`) for automated defect & spec validation.

## Tech Stack
Python · FastAPI · PyTorch (MPS) · Diffusers (SDXL / RealVisXL V4.0 / ControlNet) · SAM 2.1 · Florence-2 · Real-ESRGAN · Three.js · WebSockets

## Quick Start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# Open http://localhost:8000
```
