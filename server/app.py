"""
Satya Drishti — API Server
==========================
FastAPI backend that exposes endpoints for:
- /api/auth/* (Authentication)
- /api/analyze/media (Image & Video Forensics for Media Scanner UI)
- /api/analyze/audio (AST Synthetic Voice Detection)
- /api/analyze/text (DeBERTaV3 Coercion Detection)
- /api/analyze/multimodal (Cross-Attention Fusion)
- /api/scans/* (Scan History)
- /api/contact (Contact Form)
- /ws/live (Real-time WebSocket for Call Protection)
"""

import asyncio
import base64
import logging
import mimetypes
import os
import shutil
import tempfile
import time
import wave
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import uvicorn

from .inference_engine import engine
from .database import init_db, get_db
from .auth import get_current_user
from .models import User, Scan
from .routes.auth import router as auth_router
from .routes.scans import router as scans_router
from .routes.contact import router as contact_router
from .routes.cases import router as cases_router
from .rate_limiter import limiter
from .logging_config import setup_logging
from .config import CORS_ORIGINS, HOST, PORT, MAX_AUDIO_SIZE, MAX_VIDEO_SIZE, MAX_MEDIA_SIZE, MAX_TEXT_LENGTH

log = logging.getLogger("satyadrishti.api")
setup_logging()


# ─── Lifespan ───

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    await init_db()

    # Periodic rate limiter cleanup
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(300)
            limiter.cleanup_all()

    cleanup_task = asyncio.create_task(_cleanup_loop())
    log.info("Satya Drishti API started.")
    yield
    cleanup_task.cancel()
    log.info("Satya Drishti API shutting down.")


app = FastAPI(
    title="Satya Drishti Backend API",
    description="Multimodal Deepfake and Coercion Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(scans_router)
app.include_router(contact_router)
app.include_router(cases_router)


# ─── Helpers ───

ALLOWED_MEDIA_TYPES = {
    "image/jpeg", "image/png", "image/webp",
    "video/mp4", "video/quicktime", "video/x-msvideo",
}

MIME_TO_EXTENSION = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "audio/wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/x-wav": ".wav",
}


def _safe_extension(content_type: str) -> str:
    """Get safe file extension from MIME type (never from user filename)."""
    return MIME_TO_EXTENSION.get(content_type, mimetypes.guess_extension(content_type) or ".bin")


async def _read_with_limit(file: UploadFile, max_bytes: int, label: str) -> bytes:
    """Read an upload file with size enforcement."""
    data = await file.read()
    if len(data) > max_bytes:
        max_mb = max_bytes / (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"{label} too large. Maximum size is {max_mb:.0f}MB.")
    return data


# ─── Pydantic Models ───


class TextAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)


class MediaAnalysisResponse(BaseModel):
    verdict: str
    confidence: float
    forensic_checks: list
    raw_scores: dict


# ─── REST Endpoints ───


@app.get("/api/health")
async def health_check():
    """Service health check."""
    return {"status": "ok", "service": "Satya Drishti API"}


@app.post("/api/analyze/text")
async def analyze_text(req: Request):
    """Analyze text for coercion/scam patterns using DeBERTaV3 + LoRA."""
    limiter.check(req, limit=30, window=60, endpoint="analyze_text")

    # Accept text from either JSON body or form data
    input_text = None
    content_type = req.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await req.json()
        input_text = body.get("text")
    elif "form" in content_type:
        form = await req.form()
        input_text = form.get("text")
    else:
        # Try JSON first, fall back to form
        try:
            body = await req.json()
            input_text = body.get("text")
        except Exception:
            form = await req.form()
            input_text = form.get("text")

    if not input_text or not str(input_text).strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    input_text = str(input_text).strip()
    if len(input_text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail=f"Text too long. Maximum {MAX_TEXT_LENGTH} characters.")

    result = await engine.analyze_text(input_text)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@app.post("/api/analyze/audio")
async def analyze_audio(request: Request, file: UploadFile = File(...)):
    """Upload audio for AST synthetic voice detection."""
    limiter.check(request, limit=20, window=60, endpoint="analyze_audio")
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid audio file type")

    data = await _read_with_limit(file, MAX_AUDIO_SIZE, "Audio file")

    # Validate the audio data is actually readable
    import io
    try:
        import soundfile as sf
        sf.info(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio file. Could not read audio data.")

    result = await engine.analyze_audio(data)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@app.post("/api/analyze/video")
async def analyze_video(
    request: Request,
    file: UploadFile = File(...),
    current_user: User | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload video for forensics + two-stream deepfake detection."""
    limiter.check(request, limit=10, window=60, endpoint="analyze_video")
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid video file type")

    ext = _safe_extension(file.content_type)
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        # Stream copy with size limit
        total = 0
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_VIDEO_SIZE:
                os.remove(temp_file.name)
                raise HTTPException(status_code=413, detail=f"Video too large. Maximum size is {MAX_VIDEO_SIZE // (1024*1024)}MB.")
            temp_file.write(chunk)
        temp_path = temp_file.name

    try:
        result = await engine.analyze_video(temp_path)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        scan_id = None
        if current_user:
            scan = Scan(
                user_id=current_user.id,
                file_name=file.filename or "video",
                file_type=file.content_type,
                verdict=result.get("verdict", "inconclusive"),
                confidence=result.get("confidence", 0.0),
                forensic_data=result.get("forensic_checks", []),
                raw_scores=result.get("raw_scores", {}),
            )
            db.add(scan)
            await db.commit()
            await db.refresh(scan)
            scan_id = scan.id

        return {**result, "scan_id": scan_id}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/analyze/multimodal")
async def analyze_multimodal(
    request: Request,
    audio: UploadFile = File(None),
    video: UploadFile = File(None),
    text: str = Form(None),
):
    """All 3 modalities -> Cross-Attention Fusion -> 4-class threat assessment."""
    limiter.check(request, limit=10, window=60, endpoint="analyze_multimodal")

    # Require at least one modality
    if not audio and not video and not text:
        raise HTTPException(status_code=400, detail="At least one modality (audio, video, or text) is required")

    # Validate text length
    if text and len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail=f"Text too long. Maximum {MAX_TEXT_LENGTH} characters.")

    audio_data = None
    video_path = None

    try:
        if audio:
            if not audio.content_type or not audio.content_type.startswith("audio/"):
                raise HTTPException(status_code=400, detail="Invalid audio file type")
            audio_data = await _read_with_limit(audio, MAX_AUDIO_SIZE, "Audio file")

        if video:
            if not video.content_type or not video.content_type.startswith("video/"):
                raise HTTPException(status_code=400, detail="Invalid video file type")
            ext = _safe_extension(video.content_type)
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                shutil.copyfileobj(video.file, temp_file)
                video_path = temp_file.name

        result = await engine.analyze_multimodal(
            audio_data=audio_data,
            video_path=video_path,
            text=text,
        )

        return result
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)


@app.post("/api/analyze/media")
async def analyze_media(
    request: Request,
    file: UploadFile = File(...),
    current_user: User | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Unified media scanner endpoint for the frontend.
    Handles images (forensics pipeline) and videos (forensics + deepfake detection).
    """
    limiter.check(request, limit=15, window=60, endpoint="analyze_media")
    if file.content_type not in ALLOWED_MEDIA_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    # Stream to temp file with size limit
    ext = _safe_extension(file.content_type)
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        total = 0
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_MEDIA_SIZE:
                os.remove(temp_file.name)
                raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_MEDIA_SIZE // (1024*1024)}MB.")
            temp_file.write(chunk)
        temp_path = temp_file.name

    try:
        if file.content_type.startswith("image/"):
            # Verify the image is actually valid before sending to ML
            import cv2 as _cv2
            temp_check = _cv2.imread(temp_path)
            if temp_check is None:
                os.remove(temp_path)
                raise HTTPException(status_code=400, detail="Invalid image file. Could not decode image data.")
            result = await engine.analyze_media(temp_path)
        else:
            result = await engine.analyze_video(temp_path)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        # Save scan to database if user is authenticated
        scan_id = None
        if current_user:
            scan = Scan(
                user_id=current_user.id,
                file_name=file.filename or "media",
                file_type=file.content_type,
                verdict=result.get("verdict", "inconclusive"),
                confidence=result.get("confidence", 0.0),
                forensic_data=result.get("forensic_checks", []),
                raw_scores=result.get("raw_scores", {}),
            )
            db.add(scan)
            await db.commit()
            await db.refresh(scan)
            scan_id = scan.id

        return {**result, "scan_id": scan_id}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ─── Voice Print Enrollment Endpoints ───


@app.post("/api/voice-prints/enroll")
async def enroll_voice_print(
    request: Request,
    file: UploadFile = File(...),
    name: str = Form(...),
    relationship: str = Form("unknown"),
):
    """Enroll a family member's voice for speaker verification."""
    limiter.check(request, limit=5, window=60, endpoint="enroll_voice")

    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Please upload an audio file")

    data = await _read_with_limit(file, MAX_AUDIO_SIZE, "Audio file")
    result = await engine.enroll_voice_print(name, data, relationship)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    return result


@app.get("/api/voice-prints")
async def list_voice_prints():
    """List all enrolled voice prints."""
    verifier = await engine.get_speaker_verifier()
    if not verifier:
        return {"prints": [], "message": "Speaker verification not available"}
    return {"prints": verifier.list_enrolled()}


@app.delete("/api/voice-prints/{name}")
async def delete_voice_print(name: str):
    """Remove an enrolled voice print."""
    verifier = await engine.get_speaker_verifier()
    if not verifier:
        raise HTTPException(status_code=500, detail="Speaker verification not available")

    if verifier.remove(name):
        return {"status": "deleted", "name": name}
    raise HTTPException(status_code=404, detail=f"Voice print '{name}' not found")


# ─── WebSocket Endpoint ───


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """
    Real-time call protection WebSocket.

    Enhanced protocol:
      Client -> Server: { "type": "audio", "data": "<base64 audio chunk>" }
      Client -> Server: { "type": "text", "data": "<live transcript text>" }
      Client -> Server: { "type": "video_frame", "data": "<base64 frame>" }
      Client -> Server: { "type": "call_start" }
      Client -> Server: { "type": "call_end" }
      Server -> Client: { "type": "analysis_result", ... }
      Server -> Client: { "type": "threat_alert", "level": "warning|danger|critical", ... }
      Server -> Client: { "type": "call_summary", ... }
    """
    await websocket.accept()
    log.info("WebSocket client connected for call protection.")

    # Call state tracking
    call_state = {
        "active": False,
        "audio_analyses": [],
        "text_analyses": [],
        "threat_escalation": 0.0,       # 0-1, accumulates over call duration
        "deepfake_detections": 0,
        "coercion_detections": 0,
        "alert_level": "safe",           # safe, warning, danger, critical
        "transcript_buffer": "",
        "audio_chunks": [],              # Accumulate audio for recording
    }

    ESCALATION_DECAY = 0.95  # Threat decays slowly if no new signals

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "call_start":
                call_state["active"] = True
                call_state["threat_escalation"] = 0.0
                call_state["deepfake_detections"] = 0
                call_state["coercion_detections"] = 0
                call_state["alert_level"] = "safe"
                call_state["transcript_buffer"] = ""
                call_state["audio_chunks"] = []
                # Initialize per-call temporal tracker for cross-chunk consistency
                try:
                    from engine.audio.temporal_tracker import TemporalTracker
                    call_state["temporal_tracker"] = TemporalTracker()
                except ImportError:
                    call_state["temporal_tracker"] = None
                await websocket.send_json({"type": "call_started", "status": "monitoring"})

            elif msg_type == "audio" and data.get("data"):
                try:
                    audio_bytes = base64.b64decode(data["data"])
                    call_state["audio_chunks"].append(audio_bytes)

                    # === PIPELINE 1: 7-Layer Voice Clone Detection ===
                    audio_result = await engine.analyze_audio(audio_bytes)
                    is_spoof = False
                    audio_conf = 0

                    if "error" not in audio_result:
                        verdict = audio_result.get("verdict", "authentic")
                        is_spoof = verdict == "spoof"
                        is_uncertain = verdict == "uncertain"
                        audio_conf = audio_result.get("confidence", 0)

                        # Check ensemble probability for uncertain verdicts
                        ensemble_prob = audio_result.get("raw_scores", {}).get(
                            "ensemble_probability", audio_result.get("details", {}).get("ensemble_probability", 0)
                        )

                        if is_spoof:
                            call_state["deepfake_detections"] += 1
                            call_state["threat_escalation"] = min(1.0,
                                call_state["threat_escalation"] + (audio_conf / 100) * 0.3)
                        elif is_uncertain and ensemble_prob > 0.5:
                            # Uncertain but leaning spoof — still escalate, just less aggressively
                            call_state["threat_escalation"] = min(1.0,
                                call_state["threat_escalation"] + ensemble_prob * 0.15)
                        else:
                            call_state["threat_escalation"] *= ESCALATION_DECAY

                        ws_payload = {
                            "type": "analysis_result",
                            "modality": "audio",
                            "is_synthetic": is_spoof or (is_uncertain and ensemble_prob > 0.6),
                            "confidence": audio_conf,
                            "verdict": verdict,
                            "ensemble_probability": round(ensemble_prob, 4) if ensemble_prob else None,
                            "threat_escalation": round(call_state["threat_escalation"], 3),
                        }
                        # Include 7-layer forensic details if available
                        if "forensic_checks" in audio_result:
                            ws_payload["forensic_checks"] = audio_result["forensic_checks"]
                        if "raw_scores" in audio_result:
                            ws_payload["raw_scores"] = audio_result["raw_scores"]
                        details = audio_result.get("details", {})
                        if "layers_active" in details:
                            ws_payload["layers_active"] = details["layers_active"]
                            ws_payload["layers_total"] = details.get("layers_total", 7)
                        if "per_analyzer" in details:
                            ws_payload["per_analyzer"] = details["per_analyzer"]

                        await websocket.send_json(ws_payload)

                        # === PIPELINE 1b: Cross-Chunk Temporal Tracking ===
                        tracker = call_state.get("temporal_tracker")
                        if tracker and "raw_scores" in audio_result:
                            try:
                                # Get AST embedding for this chunk
                                emb = await engine.extract_audio_embedding(audio_bytes)
                                if emb is not None:
                                    import numpy as np
                                    emb_np = emb.cpu().numpy().flatten() if hasattr(emb, 'cpu') else np.array(emb).flatten()
                                    # Feed prosodic F0 if available
                                    f0_stats = None
                                    details = audio_result.get("details", {})
                                    per_a = details.get("per_analyzer", {})
                                    chunk_score = audio_result["raw_scores"].get("ensemble_probability")
                                    t_result = tracker.update(emb_np, f0_stats=f0_stats, chunk_score=chunk_score)
                                    if t_result.get("score", 0) > 0.6:
                                        await websocket.send_json({
                                            "type": "analysis_result",
                                            "modality": "temporal",
                                            "score": round(t_result["score"], 3),
                                            "confidence": round(t_result.get("confidence", 0), 3),
                                            "anomalies": t_result.get("anomalies", []),
                                            "chunks_analyzed": t_result.get("chunks_analyzed", 0),
                                            "threat_escalation": round(call_state["threat_escalation"], 3),
                                        })
                            except Exception as e:
                                log.debug("Temporal tracking chunk error: %s", e)

                    # === PIPELINE 2: Real-Time Transcription ===
                    transcript_result = await engine.transcribe_audio(audio_bytes)

                    if transcript_result.get("text"):
                        transcribed_text = transcript_result["text"]
                        detected_lang = transcript_result.get("language", "unknown")

                        # Send transcript to frontend (live subtitles)
                        await websocket.send_json({
                            "type": "transcript",
                            "text": transcribed_text,
                            "language": detected_lang,
                            "language_probability": transcript_result.get("language_probability", 0),
                        })

                        # AUTO-FEED transcript to coercion detector
                        text_result = await engine.analyze_text(transcribed_text)

                        if "error" not in text_result:
                            is_threat = text_result.get("verdict", "safe") != "safe"
                            text_conf = text_result.get("confidence", 0)

                            if is_threat:
                                call_state["coercion_detections"] += 1
                                call_state["threat_escalation"] = min(1.0,
                                    call_state["threat_escalation"] + (text_conf / 100) * 0.25)

                            await websocket.send_json({
                                "type": "analysis_result",
                                "modality": "text",
                                "verdict": text_result.get("verdict", "safe"),
                                "confidence": text_conf,
                                "detected_patterns": text_result.get("detected_patterns", []),
                                "language": text_result.get("language", detected_lang),
                                "threat_escalation": round(call_state["threat_escalation"], 3),
                                "auto_transcribed": True,
                            })

                    # === PIPELINE 3: Speaker Verification ===
                    verify_result = await engine.verify_speaker(audio_bytes)

                    if verify_result.get("best_match_name"):
                        await websocket.send_json({
                            "type": "speaker_verification",
                            "best_match": verify_result.get("best_match"),
                            "best_match_name": verify_result["best_match_name"],
                            "similarity": verify_result.get("similarity", 0),
                            "is_verified": verify_result.get("is_verified", False),
                            "relationship": verify_result.get("relationship", "unknown"),
                        })

                        # If caller claims to be someone but voice doesn't match
                        if not verify_result.get("is_verified") and verify_result.get("similarity", 0) < 0.15:
                            call_state["threat_escalation"] = min(1.0,
                                call_state["threat_escalation"] + 0.2)

                    # Check for escalating alerts
                    await _check_and_send_alert(websocket, call_state)

                except Exception as e:
                    log.warning("WebSocket audio pipeline error: %s", e, exc_info=True)
                    try:
                        await websocket.send_json({
                            "type": "analysis_result",
                            "modality": "audio",
                            "is_synthetic": False,
                            "confidence": 0,
                            "error": str(e),
                        })
                    except Exception:
                        pass

            elif msg_type == "text" and data.get("data"):
                try:
                    text_data = data["data"][:MAX_TEXT_LENGTH]
                    call_state["transcript_buffer"] += " " + text_data

                    result = await engine.analyze_text(text_data)

                    if "error" not in result:
                        is_threat = result.get("verdict", "safe") != "safe"
                        conf = result["confidence"] / 100

                        if is_threat:
                            call_state["coercion_detections"] += 1
                            call_state["threat_escalation"] = min(1.0,
                                call_state["threat_escalation"] + conf * 0.25)
                        else:
                            call_state["threat_escalation"] *= ESCALATION_DECAY

                        await websocket.send_json({
                            "type": "analysis_result",
                            "modality": "text",
                            "verdict": result.get("verdict", "safe"),
                            "confidence": result["confidence"],
                            "detected_patterns": result.get("detected_patterns", []),
                            "language": result.get("language", "en"),
                            "threat_escalation": round(call_state["threat_escalation"], 3),
                        })

                        await _check_and_send_alert(websocket, call_state)

                except Exception as e:
                    log.warning("WebSocket text error: %s", e)

            elif msg_type == "video_frame" and data.get("data"):
                try:
                    frame_bytes = base64.b64decode(data["data"])

                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(frame_bytes)
                        tmp_path = tmp.name

                    try:
                        result = await engine.analyze_media(tmp_path)
                        if "error" not in result:
                            verdict = result.get("verdict", "inconclusive")
                            confidence = result.get("confidence", 0)

                            if verdict == "ai-generated" and confidence > 70:
                                call_state["threat_escalation"] = min(1.0,
                                    call_state["threat_escalation"] + 0.2)

                            await websocket.send_json({
                                "type": "analysis_result",
                                "modality": "video",
                                "verdict": verdict,
                                "confidence": confidence,
                                "threat_escalation": round(call_state["threat_escalation"], 3),
                            })

                            await _check_and_send_alert(websocket, call_state)
                    finally:
                        os.remove(tmp_path)

                except Exception as e:
                    log.warning("WebSocket video frame error: %s", e)

            elif msg_type == "call_end":
                # Save recording for evidence (only if threats detected)
                recording_path = None
                if call_state["audio_chunks"] and call_state["deepfake_detections"] > 0:
                    try:
                        os.makedirs("recordings", exist_ok=True)
                        recording_path = f"recordings/call_{int(time.time())}.wav"

                        # Concatenate WAV chunks — extract raw PCM from each
                        pcm_data = bytearray()
                        sample_rate = 16000
                        sample_width = 2
                        channels = 1

                        for chunk in call_state["audio_chunks"]:
                            try:
                                import soundfile as sf
                                wf, sr = sf.read(io.BytesIO(chunk), dtype="int16")
                                pcm_data.extend(wf.tobytes())
                                sample_rate = sr
                            except Exception:
                                pass

                        if pcm_data:
                            with wave.open(recording_path, "wb") as wf:
                                wf.setnchannels(channels)
                                wf.setsampwidth(sample_width)
                                wf.setframerate(sample_rate)
                                wf.writeframes(bytes(pcm_data))
                            log.info("Call recording saved: %s", recording_path)
                    except Exception as e:
                        log.warning("Failed to save call recording: %s", e)
                        recording_path = None

                call_state["audio_chunks"] = []

                # Generate call summary
                summary = {
                    "type": "call_summary",
                    "deepfake_detections": call_state["deepfake_detections"],
                    "coercion_detections": call_state["coercion_detections"],
                    "peak_threat_level": call_state["alert_level"],
                    "final_threat_score": round(call_state["threat_escalation"], 3),
                    "recommendation": _get_recommendation(call_state),
                }
                if recording_path:
                    summary["recording_path"] = recording_path

                await websocket.send_json(summary)
                call_state["active"] = False

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except Exception as e:
        log.info("WebSocket disconnected: %s", e)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


async def _check_and_send_alert(websocket: WebSocket, state: dict):
    """Send escalating threat alerts based on accumulated evidence."""
    escalation = state["threat_escalation"]
    new_level = "safe"

    if escalation >= 0.8:
        new_level = "critical"
    elif escalation >= 0.6:
        new_level = "danger"
    elif escalation >= 0.3:
        new_level = "warning"

    if new_level != state["alert_level"] and new_level != "safe":
        state["alert_level"] = new_level

        alert_messages = {
            "warning": "Potential suspicious activity detected. Stay cautious.",
            "danger": "Multiple threat indicators detected. Consider ending the call.",
            "critical": "HIGH THREAT: Voice cloning and/or coercion patterns detected. End the call immediately.",
        }

        await websocket.send_json({
            "type": "threat_alert",
            "level": new_level,
            "message": alert_messages.get(new_level, ""),
            "deepfake_count": state["deepfake_detections"],
            "coercion_count": state["coercion_detections"],
            "threat_score": round(escalation, 3),
        })


def _get_recommendation(state: dict) -> str:
    """Generate end-of-call safety recommendation."""
    d = state["deepfake_detections"]
    c = state["coercion_detections"]

    if d > 0 and c > 0:
        return (
            f"WARNING: {d} synthetic voice detection(s) and {c} coercion pattern(s) detected. "
            f"This call shows strong indicators of an AI-enabled scam. "
            f"Do NOT share any personal or financial information. "
            f"Report this number to your local cyber crime cell."
        )
    elif d > 0:
        return (
            f"CAUTION: {d} synthetic voice detection(s). The caller's voice may be AI-generated. "
            f"Verify the caller's identity through a different channel before taking any action."
        )
    elif c > 0:
        return (
            f"CAUTION: {c} coercion/manipulation pattern(s) detected. "
            f"The caller used pressure tactics. Take time to verify any claims independently."
        )
    else:
        return "No threats detected during this call."


if __name__ == "__main__":
    uvicorn.run("server.app:app", host=HOST, port=PORT, reload=True)
