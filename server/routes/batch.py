"""
Satya Drishti — Batch Analysis API
====================================
POST /api/batch/analyze     — Upload ZIP file → analyze all media → return batch report
GET  /api/batch/{batch_id}  — Get batch analysis status/results
"""

import asyncio
import io
import logging
import os
import tempfile
import uuid
import zipfile
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Query

logger = logging.getLogger("satyadrishti.batch")
router = APIRouter(prefix="/api/batch", tags=["batch"])

# In-memory batch job storage (for MVP; use DB for production)
_batch_jobs: dict = {}

# Supported media types for batch analysis
SUPPORTED_EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"},
    "audio": {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".wma"},
    "video": {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"},
    "document": {".pdf"},
}

ALL_SUPPORTED = set()
for exts in SUPPORTED_EXTENSIONS.values():
    ALL_SUPPORTED.update(exts)

MAX_ZIP_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_FILES_PER_BATCH = 50


@router.post("/analyze")
async def batch_analyze(
    file: UploadFile = File(...),
    parallel: int = Query(default=3, ge=1, le=10, description="Max parallel analyses"),
):
    """
    Upload a ZIP file containing media files for batch analysis.

    Returns a batch_id that can be used to poll for results.
    Files are analyzed in parallel (up to `parallel` concurrent tasks).
    """
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "File must be a ZIP archive (.zip)")

    # Read the ZIP into memory
    content = await file.read()
    if len(content) > MAX_ZIP_SIZE:
        raise HTTPException(413, f"ZIP file too large ({len(content) / 1024 / 1024:.1f} MB). Max: {MAX_ZIP_SIZE / 1024 / 1024:.0f} MB")

    # Validate ZIP
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        raise HTTPException(400, "Invalid ZIP file")

    # List files and filter supported types
    entries = []
    total_uncompressed = 0
    MAX_UNCOMPRESSED = 500 * 1024 * 1024  # 500 MB max uncompressed (zip bomb protection)
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename
        # Zip slip protection: reject entries with path traversal
        if name.startswith("/") or ".." in name or name.startswith("\\"):
            continue
        # Use only the basename to prevent directory traversal
        basename = os.path.basename(name)
        if not basename:
            continue
        total_uncompressed += info.file_size
        if total_uncompressed > MAX_UNCOMPRESSED:
            zf.close()
            raise HTTPException(400, "ZIP contents too large when uncompressed (possible zip bomb)")
        ext = os.path.splitext(basename)[1].lower()
        if ext in ALL_SUPPORTED:
            # Determine modality
            for modality, exts in SUPPORTED_EXTENSIONS.items():
                if ext in exts:
                    entries.append({
                        "filename": name,
                        "basename": basename,
                        "size": info.file_size,
                        "modality": modality,
                        "ext": ext,
                    })
                    break

    if not entries:
        zf.close()
        raise HTTPException(400, "No supported media files found in ZIP")

    if len(entries) > MAX_FILES_PER_BATCH:
        zf.close()
        raise HTTPException(400, f"Too many files ({len(entries)}). Max: {MAX_FILES_PER_BATCH}")

    # Create batch job
    batch_id = str(uuid.uuid4())[:12]
    job = {
        "id": batch_id,
        "status": "processing",
        "total_files": len(entries),
        "completed": 0,
        "files": [],
        "results": [],
        "errors": [],
    }

    for entry in entries:
        job["files"].append({
            "filename": entry["filename"],
            "modality": entry["modality"],
            "status": "pending",
        })

    _batch_jobs[batch_id] = job

    # Start processing in background
    asyncio.create_task(_process_batch(batch_id, zf, entries, parallel))

    return {
        "batch_id": batch_id,
        "status": "processing",
        "total_files": len(entries),
        "files": [
            {"filename": e["filename"], "modality": e["modality"]}
            for e in entries
        ],
        "poll_url": f"/api/batch/{batch_id}",
    }


@router.get("/{batch_id}")
async def get_batch_status(batch_id: str):
    """Get the status and results of a batch analysis job."""
    job = _batch_jobs.get(batch_id)
    if not job:
        raise HTTPException(404, f"Batch job {batch_id} not found")

    return {
        "id": job["id"],
        "status": job["status"],
        "total_files": job["total_files"],
        "completed": job["completed"],
        "files": job["files"],
        "results": job["results"],
        "errors": job["errors"],
        "summary": _generate_summary(job) if job["status"] == "completed" else None,
    }


async def _process_batch(batch_id: str, zf: zipfile.ZipFile, entries: list, parallel: int):
    """Process all files in the batch with bounded parallelism."""
    from server.inference_engine import get_engine

    job = _batch_jobs.get(batch_id)
    if not job:
        return

    semaphore = asyncio.Semaphore(parallel)

    async def process_file(idx: int, entry: dict):
        async with semaphore:
            filename = entry["filename"]
            modality = entry["modality"]

            job["files"][idx]["status"] = "processing"

            try:
                # Extract to temp file
                data = zf.read(filename)
                ext = entry["ext"]
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name

                try:
                    engine = get_engine()
                    result = await _analyze_file(engine, tmp_path, modality)
                    job["results"].append({
                        "filename": filename,
                        "modality": modality,
                        **result,
                    })
                    job["files"][idx]["status"] = "completed"
                    job["files"][idx]["verdict"] = result.get("verdict", "unknown")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

            except Exception as e:
                logger.error("Batch analysis error for %s: %s", filename, e)
                job["errors"].append({
                    "filename": filename,
                    "error": str(e),
                })
                job["files"][idx]["status"] = "error"
                job["files"][idx]["error"] = str(e)

            job["completed"] += 1

    # Run all files with semaphore-bounded parallelism
    tasks = [process_file(i, entry) for i, entry in enumerate(entries)]
    await asyncio.gather(*tasks)

    zf.close()
    job["status"] = "completed"
    logger.info("Batch %s completed: %d/%d files processed",
                batch_id, job["completed"], job["total_files"])


async def _analyze_file(engine, file_path: str, modality: str) -> dict:
    """Analyze a single file based on its modality."""
    if modality == "image":
        forensics = await engine.get_forensics_model()
        if forensics:
            return forensics.analyze(file_path)
        return {"verdict": "error", "confidence": 0, "error": "Forensics engine not available"}

    elif modality == "audio":
        result = await engine.analyze_audio_full(file_path)
        return result

    elif modality == "video":
        result = await engine.analyze_video_full(file_path)
        return result

    elif modality == "document":
        try:
            from engine.document_forensics import DocumentForensicsDetector
            detector = DocumentForensicsDetector()
            return detector.analyze(file_path)
        except ImportError:
            return {"verdict": "error", "confidence": 0, "error": "Document forensics not available"}

    return {"verdict": "unsupported", "confidence": 0}


def _generate_summary(job: dict) -> dict:
    """Generate a summary of batch analysis results."""
    results = job["results"]
    if not results:
        return {"total": 0, "ai_detected": 0, "authentic": 0, "uncertain": 0}

    ai_verdicts = {"ai-generated", "spoof", "manipulated", "deepfake", "suspicious"}
    real_verdicts = {"authentic", "bonafide", "real"}

    ai_count = sum(1 for r in results if r.get("verdict", "").lower() in ai_verdicts)
    real_count = sum(1 for r in results if r.get("verdict", "").lower() in real_verdicts)
    uncertain_count = len(results) - ai_count - real_count

    return {
        "total": len(results),
        "ai_detected": ai_count,
        "authentic": real_count,
        "uncertain": uncertain_count,
        "error_count": len(job["errors"]),
        "detection_rate": round(ai_count / len(results) * 100, 1) if results else 0,
    }
