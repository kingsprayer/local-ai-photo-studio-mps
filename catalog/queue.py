"""Disk-persisted job queue with automatic recovery and retry management."""

import os
import json
import uuid
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)

PENDING_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads", "pending")


class JobQueue:
    def __init__(self, pending_dir: str = PENDING_DIR):
        self.pending_dir = pending_dir
        os.makedirs(self.pending_dir, exist_ok=True)
        self.memory_jobs: Dict[str, Dict[str, Any]] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.listeners: List[Callable[[Dict[str, Any]], None]] = []
        self.load_from_disk()

    def load_from_disk(self):
        """Load and recover pending jobs from disk on startup (max 2 retries)."""
        if not os.path.exists(self.pending_dir):
            return

        for fname in os.listdir(self.pending_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(self.pending_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    job_id = data.get("id")
                    if not job_id:
                        continue
                    
                    status = data.get("status", "queued")
                    retries = data.get("retries", 0)

                    # Resume unfinished or interrupted jobs
                    if status in ["processing", "queued"]:
                        if retries < 2:
                            data["retries"] = retries + 1
                            data["status"] = "queued"
                            data["log"] = data.get("log", []) + [f"Resumed from disk restart (retry {data['retries']}/2)"]
                            self.save_job_to_disk(data)
                            logger.info(f"Recovered pending job {job_id} (retry {data['retries']}/2)")
                        else:
                            data["status"] = "failed"
                            data["error"] = "Exceeded maximum restart retries (2/2)."
                            self.save_job_to_disk(data)

                    self.memory_jobs[job_id] = data
                except Exception as e:
                    logger.error(f"Error loading pending job {fname}: {e}")

    def save_job_to_disk(self, job: Dict[str, Any]):
        """Persist job state to uploads/pending/<job_id>.json."""
        job_id = job["id"]
        fpath = os.path.join(self.pending_dir, f"{job_id}.json")
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(job, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist job {job_id} to disk: {e}")

    def create_job(
        self,
        image_path: str,
        job_name: str,
        mode: str,
        channels: List[str],
        attributes: Dict[str, Any],
        options: Dict[str, Any],
        batch_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create and persist a new job."""
        job_id = str(uuid.uuid4())[:12]
        now = time.time()
        job = {
            "id": job_id,
            "name": job_name,
            "batch_id": batch_id,
            "created_at": now,
            "updated_at": now,
            "status": "queued",
            "progress": 0,
            "current_step": "Waiting in queue...",
            "retries": 0,
            "image_path": image_path,
            "mode": mode,
            "channels": channels,
            "attributes": attributes,
            "options": options,
            "results": {},
            "log": ["Job registered and queued."]
        }
        self.memory_jobs[job_id] = job
        self.save_job_to_disk(job)
        self.queue.put_nowait(job_id)
        self.notify_listeners(job)
        return job

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.memory_jobs.get(job_id)

    def list_jobs(self, batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        jobs = list(self.memory_jobs.values())
        if batch_id:
            jobs = [j for j in jobs if j.get("batch_id") == batch_id]
        jobs.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return jobs

    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        step: Optional[str] = None,
        log_message: Optional[str] = None,
        results: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Update job progress and state, persist to disk, and notify listeners."""
        job = self.memory_jobs.get(job_id)
        if not job:
            return
        job["updated_at"] = time.time()
        if status is not None:
            job["status"] = status
        if progress is not None:
            job["progress"] = progress
        if step is not None:
            job["current_step"] = step
        if log_message:
            job["log"].append(f"[{time.strftime('%H:%M:%S')}] {log_message}")
        if results is not None:
            job["results"] = results
        if error is not None:
            job["error"] = error

        self.save_job_to_disk(job)
        self.notify_listeners(job)

    def add_listener(self, listener: Callable[[Dict[str, Any]], None]):
        self.listeners.append(listener)

    def remove_listener(self, listener: Callable[[Dict[str, Any]], None]):
        if listener in self.listeners:
            self.listeners.remove(listener)

    def notify_listeners(self, job: Dict[str, Any]):
        for listener in self.listeners:
            try:
                listener(job)
            except Exception:
                pass


# Global singleton queue
job_queue = JobQueue()
