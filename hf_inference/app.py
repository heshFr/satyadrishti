"""
Satya Drishti — HuggingFace Spaces Inference Worker
=====================================================
Standalone FastAPI service that wraps the full ML engine pipeline.
Deployed on HF Spaces (16GB RAM, free tier) to handle all GPU/CPU inference.

The API Gateway (Render) forwards media files here for analysis.
This service has NO database, NO auth — it only does inference.
"""

import asyncio
import io
import logging
import os
import sys
import tempfile
import time
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add project root to path so engine/ imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("satyadrishti.inference")

# ── Lazy import the local inference engine ──
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        # Set offline mode to avoid HF rate limits
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

        from server.inference_engine import InferenceEngine
        _engine = InferenceEngine()
        log.info("Inference engine initialized on device: %s", _engine.device)
    return _engine


# ── Auth token for gateway-to-inference requests ──
INFERENCE_SECRET = os.environ.get("INFERENCE_SECRET", "")


def verify_token(request: Request):
    """Simple shared-secret auth between gateway and inference worker."""
    if not INFERENCE_SECRET:
        return  # No secret set = open (dev mode)
    token = request.headers.get("X-Inference-Token", "")
    if token != INFERENCE_SECRET:
        raise HTTPException(status_code=403, detail="Invalid inference token")


# ── FastAPI App ──

app = FastAPI(
    title="Satya Drishti Inference Worker",
    description="ML inference service for deepfake & coercion detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check — also reports which models are available."""
    engine = get_engine()
    return {
        "status": "ok",
        "device": str(engine.device),
        "models_loaded": {k: v is not None for k, v in engine.models.items()},
    }


class TextRequest(BaseModel):
    text: str


@app.post("/infer/text")
async def infer_text(req: TextRequest, request: Request):
    """Analyze text for coercion/manipulation."""
    verify_token(request)
    engine = get_engine()
    start = time.time()
    result = await engine.analyze_text(req.text)
    result["inference_ms"] = round((time.time() - start) * 1000)
    return result


@app.post("/infer/audio")
async def infer_audio(request: Request, file: UploadFile = File(...)):
    """Analyze audio with the 12-layer ensemble."""
    verify_token(request)
    engine = get_engine()
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "Empty audio file")
    if len(data) > 100 * 1024 * 1024:
        raise HTTPException(413, "Audio file too large (max 100MB)")

    start = time.time()
    result = await engine.analyze_audio(data)
    result["inference_ms"] = round((time.time() - start) * 1000)
    return result


@app.post("/infer/media")
async def infer_media(request: Request, file: UploadFile = File(...)):
    """Analyze image with the forensics pipeline."""
    verify_token(request)
    engine = get_engine()
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "Empty file")
    if len(data) > 100 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 100MB)")

    # Write to temp file (forensics pipeline needs a file path)
    ext = ".jpg"
    ct = file.content_type or ""
    if "png" in ct:
        ext = ".png"
    elif "webp" in ct:
        ext = ".webp"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        start = time.time()
        result = await engine.analyze_media(tmp_path)
        result["inference_ms"] = round((time.time() - start) * 1000)
        return result
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/infer/video")
async def infer_video(request: Request, file: UploadFile = File(...)):
    """Analyze video with the 10-engine ensemble."""
    verify_token(request)
    engine = get_engine()

    ext = ".mp4"
    ct = file.content_type or ""
    if "avi" in ct:
        ext = ".avi"
    elif "matroska" in ct or "mkv" in ct:
        ext = ".mkv"
    elif "webm" in ct:
        ext = ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        total = 0
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > 500 * 1024 * 1024:
                os.remove(tmp.name)
                raise HTTPException(413, "Video too large (max 500MB)")
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        start = time.time()
        result = await engine.analyze_video(tmp_path)
        result["inference_ms"] = round((time.time() - start) * 1000)
        return result
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/infer/multimodal")
async def infer_multimodal(
    request: Request,
    audio: UploadFile = File(None),
    video: UploadFile = File(None),
    text: str = Form(None),
):
    """Multimodal fusion analysis."""
    verify_token(request)
    engine = get_engine()

    if not audio and not video and not text:
        raise HTTPException(400, "At least one modality required")

    audio_data = None
    video_path = None

    try:
        if audio:
            audio_data = await audio.read()
        if video:
            video_bytes = await video.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_bytes)
                video_path = tmp.name

        start = time.time()
        result = await engine.analyze_multimodal(
            audio_data=audio_data,
            video_path=video_path,
            text=text,
        )
        result["inference_ms"] = round((time.time() - start) * 1000)
        return result
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)


@app.post("/infer/transcribe")
async def infer_transcribe(request: Request, file: UploadFile = File(...)):
    """Transcribe audio to text."""
    verify_token(request)
    engine = get_engine()
    data = await file.read()
    result = await engine.transcribe_audio(data)
    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))  # HF Spaces default port
    uvicorn.run(app, host="0.0.0.0", port=port)
