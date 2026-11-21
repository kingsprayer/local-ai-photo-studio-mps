#!/usr/bin/env python3
"""Batch processing CLI for SAADAA Local AI Product Photo Studio.

Accepts any mix of folders and files, processes them recursively, assigns
each image a job named <folder>/<file>, tracks batch progress, and writes
a comprehensive batch_report.json.
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from catalog.channels import CHANNEL_FORMATS
from catalog.pipeline import process_catalog_job
from catalog.vision import extract_attributes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("saadaa.batch")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def find_images(inputs: List[str]) -> List[Dict[str, str]]:
    """Scan mixed inputs (files and directories) recursively for image assets."""
    found = []
    for item in inputs:
        path = Path(item).resolve()
        if not path.exists():
            logger.warning(f"Input path does not exist: {path}")
            continue

        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                parent_name = path.parent.name or "root"
                job_name = f"{parent_name}/{path.name}"
                found.append({"file_path": str(path), "job_name": job_name, "folder": parent_name})
        elif path.is_dir():
            for root, _, files in os.walk(path):
                for f in sorted(files):
                    fpath = Path(root) / f
                    if fpath.suffix.lower() in SUPPORTED_EXTENSIONS:
                        rel_folder = fpath.parent.relative_to(path)
                        folder_label = path.name if str(rel_folder) == "." else f"{path.name}/{rel_folder}"
                        job_name = f"{folder_label}/{f}"
                        found.append({"file_path": str(fpath), "job_name": job_name, "folder": folder_label})

    return found


def run_batch(
    inputs: List[str],
    mode: str = "packshot",
    channels: List[str] = None,
    quality_preset: str = "standard",
    batch_prefix: str = None,
    mock_mode: bool = False,
    report_file: str = "batch_report.json"
):
    if channels is None:
        channels = ["amazon", "myntra", "shopify"]

    images = find_images(inputs)
    total_count = len(images)
    if total_count == 0:
        logger.error("No valid apparel images found in specified input paths.")
        sys.exit(1)

    print(f"\n=================================================================")
    print(f"   SAADAA Studio — Batch Catalog Generation Pipeline            ")
    print(f"   Found {total_count} images | Mode: {mode.upper()} | Channels: {', '.join(channels)}")
    print(f"=================================================================\n")

    start_time = time.time()
    results_list = []
    success_count = 0
    fail_count = 0

    for idx, item in enumerate(images, 1):
        fpath = item["file_path"]
        job_name = item["job_name"]
        if batch_prefix:
            job_name = f"{batch_prefix}_{job_name}"

        print(f"[{idx}/{total_count}] Processing: {job_name} ...")
        t0 = time.time()

        try:
            with Image.open(fpath) as img:
                source_img = img.convert("RGB")

            # Extract or analyze attributes
            attrs = {}
            if not mock_mode:
                try:
                    attrs = extract_attributes(source_img)
                except Exception:
                    pass

            job_result = process_catalog_job(
                image_input=source_img,
                job_name=job_name,
                mode=mode,
                channels=channels,
                attributes=attrs,
                options={
                    "quality_preset": quality_preset,
                    "keep_mode": "upper"
                },
                mock_mode=mock_mode
            )

            dur = round(time.time() - t0, 2)
            success_count += 1
            fidelity = job_result.get("pixel_fidelity", 100.0)
            qa_verdict = job_result.get("qa", {}).get("verdict", "PASS")

            print(f"   ✓ Finished in {dur}s | Fidelity: {fidelity}% | QA: {qa_verdict}")
            for ch_key, ch_val in job_result.get("channels", {}).items():
                badge = ch_val.get("compliance", {}).get("badge", "✓")
                status = ch_val.get("compliance", {}).get("status", "pass")
                print(f"     - {ch_key:<15} [{badge} {status.upper()}] -> {ch_val.get('relative_url')}")

            results_list.append({
                "job_name": job_name,
                "source_file": fpath,
                "status": "success",
                "duration_seconds": dur,
                "pixel_fidelity": fidelity,
                "qa": job_result.get("qa", {}),
                "channels": {
                    k: {
                        "dimensions": v.get("dimensions"),
                        "file_path": v.get("file_path"),
                        "compliance": v.get("compliance")
                    } for k, v in job_result.get("channels", {}).items()
                }
            })

        except Exception as e:
            fail_count += 1
            dur = round(time.time() - t0, 2)
            logger.exception(f"Failed processing {job_name}: {e}")
            results_list.append({
                "job_name": job_name,
                "source_file": fpath,
                "status": "failed",
                "duration_seconds": dur,
                "error": str(e)
            })

    total_duration = round(time.time() - start_time, 2)
    avg_fidelity = (
        round(sum(r.get("pixel_fidelity", 0) for r in results_list if r.get("status") == "success") / max(1, success_count), 2)
    )

    # Compile final batch report
    batch_report = {
        "batch_id": f"batch_{int(time.time())}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode,
        "quality_preset": quality_preset,
        "channels": channels,
        "mock_mode": mock_mode,
        "summary": {
            "total_images": total_count,
            "success": success_count,
            "failed": fail_count,
            "success_rate_pct": round((success_count / total_count) * 100, 1),
            "total_duration_seconds": total_duration,
            "avg_seconds_per_image": round(total_duration / max(1, total_count), 2),
            "average_garment_fidelity": avg_fidelity
        },
        "results": results_list
    }

    report_path = os.path.abspath(report_file)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(batch_report, f, indent=2)

    print(f"\n=================================================================")
    print(f"   Batch Complete! Success: {success_count}/{total_count} ({batch_report['summary']['success_rate_pct']}%)")
    print(f"   Total Time: {total_duration}s | Avg Fidelity: {avg_fidelity}%")
    print(f"   Detailed Report written to: {report_path}")
    print(f"=================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="SAADAA Local AI Product Studio Batch CLI")
    parser.add_argument("inputs", nargs="+", help="Image files and/or directories to process")
    parser.add_argument("--mode", choices=["packshot", "scene", "model"], default="packshot", help="Generation mode")
    parser.add_argument("--channels", default="amazon,myntra,shopify", help="Comma-separated channel formats")
    parser.add_argument("--quality", choices=["draft", "standard", "best"], default="standard", help="Quality preset")
    parser.add_argument("--prefix", default=None, help="Optional batch naming prefix")
    parser.add_argument("--mock", action="store_true", help="Run with mock diffusion for instantaneous verification")
    parser.add_argument("--report", default="batch_report.json", help="Path to write batch_report.json")

    args = parser.parse_args()
    channel_list = [ch.strip() for ch in args.channels.split(",") if ch.strip() in CHANNEL_FORMATS]
    if not channel_list:
        channel_list = ["amazon", "myntra", "shopify"]

    run_batch(
        inputs=args.inputs,
        mode=args.mode,
        channels=channel_list,
        quality_preset=args.quality,
        batch_prefix=args.prefix,
        mock_mode=args.mock,
        report_file=args.report
    )


if __name__ == "__main__":
    main()
