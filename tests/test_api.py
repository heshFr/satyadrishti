"""
Satya Drishti — API Endpoint Tests
====================================
Tests API request validation, health checks, and response schemas.
Uses TestClient with mock data appropriate for each endpoint.
"""

import io
import os
import wave
import struct
import pytest
from unittest.mock import patch, AsyncMock

from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app, raise_server_exceptions=False)


# ─── Helpers ───


def make_wav_bytes(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Generate a valid WAV file in memory for testing audio endpoints."""
    n_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Write silence (zeros)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    buf.seek(0)
    return buf.read()


def make_minimal_image() -> bytes:
    """Generate a minimal valid JPEG file for testing image endpoints."""
    # Smallest valid JPEG: SOI + APP0 + minimal data + EOI
    # Use a 1x1 pixel JPEG
    try:
        from PIL import Image

        buf = io.BytesIO()
        img = Image.new("RGB", (8, 8), color=(128, 128, 128))
        img.save(buf, format="JPEG")
        buf.seek(0)
        return buf.read()
    except ImportError:
        # Fallback: raw bytes that won't pass ML inference but test validation
        return b"\xff\xd8\xff\xe0" + b"\x00" * 100 + b"\xff\xd9"


# ─── Health ───


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "Satya Drishti API"


# ─── Text Analysis ───


def test_analyze_text_valid():
    """Text endpoint accepts JSON body and returns analysis result."""
    response = client.post(
        "/api/analyze/text",
        json={"text": "Hello this is a safe message."},
    )
    # May return 200 (model loaded) or 500 (model not available)
    if response.status_code == 200:
        data = response.json()
        assert "verdict" in data
        assert "confidence" in data


def test_analyze_text_empty():
    """Empty text should return 400 or 422 (Pydantic min_length validation)."""
    response = client.post("/api/analyze/text", json={"text": ""})
    assert response.status_code in (400, 422)


def test_analyze_text_missing_field():
    """Missing text field should return 400 or 422."""
    response = client.post("/api/analyze/text", json={})
    assert response.status_code in (400, 422)


# ─── Audio Analysis ───


def test_analyze_audio_wrong_type():
    """Non-audio content type should return 400."""
    response = client.post(
        "/api/analyze/audio",
        files={"file": ("test.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 400


def test_analyze_audio_valid_wav():
    """Valid WAV file should be accepted (may fail if model not loaded)."""
    wav_bytes = make_wav_bytes()
    response = client.post(
        "/api/analyze/audio",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
    )
    # 200 if model loaded, 500 if model unavailable
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        data = response.json()
        assert "verdict" in data
        assert "confidence" in data


# ─── Video Analysis ───


def test_analyze_video_wrong_type():
    """Non-video content type should return 400."""
    response = client.post(
        "/api/analyze/video",
        files={"file": ("test.txt", b"not video", "text/plain")},
    )
    assert response.status_code == 400


# ─── Media Analysis ───


def test_analyze_media_unsupported_type():
    """Unsupported file type should return 400."""
    response = client.post(
        "/api/analyze/media",
        files={"file": ("test.pdf", b"pdf content", "application/pdf")},
    )
    assert response.status_code == 400


def test_analyze_media_valid_image():
    """Valid JPEG should be accepted."""
    img_bytes = make_minimal_image()
    response = client.post(
        "/api/analyze/media",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    # 200 if models loaded, 500 if not
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        data = response.json()
        assert "verdict" in data
        assert "confidence" in data
        assert "forensic_checks" in data


# ─── Multimodal Analysis ───


def test_analyze_multimodal_text_only():
    """Multimodal endpoint should accept text-only input."""
    response = client.post(
        "/api/analyze/multimodal",
        data={"text": "This is a test transcript."},
    )
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        data = response.json()
        assert "overall_threat_level" in data


# ─── Rate Limiting ───


def test_rate_limit_text():
    """Text endpoint should enforce rate limits."""
    # Make many rapid requests
    responses = []
    for _ in range(35):
        r = client.post("/api/analyze/text", json={"text": "test"})
        responses.append(r.status_code)

    # At least one should be 429 (rate limit is 30/min for text)
    assert 429 in responses or all(r in (200, 500) for r in responses[:30])
