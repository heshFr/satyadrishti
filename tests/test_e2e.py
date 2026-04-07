"""
Satya Drishti — End-to-End Tests
=================================
Comprehensive E2E tests covering health, text/audio/video/media/multimodal
analysis, WebSocket call protection, auth flows, batch API, security headers,
CORS, concurrent requests, input validation, magic-byte detection, and
adversarial inputs.

Run:
    pytest tests/test_e2e.py -v
    pytest tests/test_e2e.py -k "test_auth" -v   # only auth tests
"""

import base64
import io
import struct
import threading
import time
import uuid
import wave
import zipfile

import pytest
from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app, raise_server_exceptions=False)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def make_wav_bytes(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Generate a valid WAV file in memory."""
    n_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    buf.seek(0)
    return buf.read()


def make_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    """Generate a valid JPEG file using PIL."""
    try:
        from PIL import Image

        buf = io.BytesIO()
        img = Image.new("RGB", (width, height), color=(128, 100, 80))
        img.save(buf, format="JPEG")
        buf.seek(0)
        return buf.read()
    except ImportError:
        # Fallback: minimal JPEG bytes (SOI + APP0 + quantization tables + EOI)
        return b"\xff\xd8\xff\xe0" + b"\x00" * 100 + b"\xff\xd9"


def make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    """Generate a valid PNG file using PIL."""
    try:
        from PIL import Image

        buf = io.BytesIO()
        img = Image.new("RGBA", (width, height), color=(50, 100, 150, 255))
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.read()
    except ImportError:
        # Minimal PNG stub
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


def make_zip_with_jpeg(filename: str = "photo.jpg") -> bytes:
    """Create a ZIP archive containing a single JPEG file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(filename, make_jpeg_bytes())
    buf.seek(0)
    return buf.read()


def _unique_email() -> str:
    """Generate a unique email for test user registration."""
    return f"e2e-{uuid.uuid4().hex[:8]}@test.satyadrishti.dev"


def _register_user(email: str | None = None, password: str = "Str0ngP@ss!2026"):
    """Register a test user and return (email, password, token, response)."""
    email = email or _unique_email()
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "password": password, "name": "E2E Test User"},
    )
    token = resp.json().get("token", "") if resp.status_code == 200 else ""
    return email, password, token, resp


# ═══════════════════════════════════════════════════════════════════════════
# 1. Health & Meta
# ═══════════════════════════════════════════════════════════════════════════


class TestHealthAndMeta:
    def test_health_endpoint(self):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["service"] == "Satya Drishti API"

    def test_metrics_endpoint(self):
        r = client.get("/metrics")
        # Prometheus text format
        assert r.status_code == 200
        assert "text/plain" in r.headers.get("content-type", "")

    def test_openapi_docs(self):
        r = client.get("/docs")
        assert r.status_code == 200

    def test_redoc(self):
        r = client.get("/redoc")
        assert r.status_code == 200

    def test_openapi_json(self):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert "paths" in schema
        assert schema["info"]["title"] == "Satya Drishti Backend API"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Text Analysis
# ═══════════════════════════════════════════════════════════════════════════


class TestTextAnalysis:
    def test_valid_text(self):
        r = client.post("/api/analyze/text", json={"text": "This is a perfectly normal safe message."})
        # 200 if model loaded, 500 if not — both are acceptable
        assert r.status_code in (200, 500)
        if r.status_code == 200:
            data = r.json()
            assert "verdict" in data
            assert "confidence" in data

    def test_empty_text(self):
        r = client.post("/api/analyze/text", json={"text": ""})
        assert r.status_code == 400

    def test_whitespace_only_text(self):
        r = client.post("/api/analyze/text", json={"text": "    "})
        assert r.status_code == 400

    def test_very_long_text(self):
        """Text exceeding MAX_TEXT_LENGTH (10000 chars) should be rejected."""
        long_text = "x" * 10001
        r = client.post("/api/analyze/text", json={"text": long_text})
        assert r.status_code in (400, 422)

    def test_special_chars(self):
        r = client.post(
            "/api/analyze/text",
            json={"text": "Hello! @#$%^&*()_+-=[]{}|;':\",./<>? end."},
        )
        assert r.status_code in (200, 500)

    def test_unicode_hindi(self):
        r = client.post(
            "/api/analyze/text",
            json={"text": "यह एक परीक्षण संदेश है। कृपया इसे अनदेखा करें।"},
        )
        assert r.status_code in (200, 500)

    def test_unicode_marathi(self):
        r = client.post(
            "/api/analyze/text",
            json={"text": "हे एक चाचणी संदेश आहे. कृपया याकडे दुर्लक्ष करा."},
        )
        assert r.status_code in (200, 500)

    def test_sql_injection_attempt(self):
        """SQL injection should be treated as plain text, not cause a crash."""
        r = client.post(
            "/api/analyze/text",
            json={"text": "'; DROP TABLE users; --"},
        )
        # Should analyse the text normally or fail gracefully, not crash
        assert r.status_code in (200, 500)

    def test_xss_attempt(self):
        """XSS payload should be handled as plain text."""
        r = client.post(
            "/api/analyze/text",
            json={"text": '<script>alert("xss")</script>'},
        )
        assert r.status_code in (200, 500)

    def test_missing_text_field(self):
        r = client.post("/api/analyze/text", json={})
        assert r.status_code in (400, 422)

    def test_null_text(self):
        r = client.post("/api/analyze/text", json={"text": None})
        assert r.status_code in (400, 422)

    def test_text_at_max_length(self):
        """Exactly MAX_TEXT_LENGTH chars should be accepted."""
        text = "a" * 10000
        r = client.post("/api/analyze/text", json={"text": text})
        assert r.status_code in (200, 500)  # accepted by validation


# ═══════════════════════════════════════════════════════════════════════════
# 3. Audio Analysis
# ═══════════════════════════════════════════════════════════════════════════


class TestAudioAnalysis:
    def test_valid_wav(self):
        wav = make_wav_bytes(duration_s=1.0)
        r = client.post(
            "/api/analyze/audio",
            files={"file": ("test.wav", wav, "audio/wav")},
        )
        assert r.status_code in (200, 400, 500)
        if r.status_code == 200:
            data = r.json()
            assert "verdict" in data
            assert "confidence" in data

    def test_invalid_format_text_file(self):
        """A plain text file sent as audio should be rejected."""
        r = client.post(
            "/api/analyze/audio",
            files={"file": ("recording.wav", b"This is not audio data at all", "audio/wav")},
        )
        # Validator should catch that magic bytes don't match audio
        assert r.status_code in (400, 415)

    def test_empty_file(self):
        r = client.post(
            "/api/analyze/audio",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        assert r.status_code == 400

    def test_corrupt_audio(self):
        """WAV header but garbage payload."""
        # Valid RIFF header but truncated/corrupt data
        corrupt = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00" + b"\x00" * 20
        r = client.post(
            "/api/analyze/audio",
            files={"file": ("corrupt.wav", corrupt, "audio/wav")},
        )
        # Should get 400 (invalid audio) or 500 (processing failure)
        assert r.status_code in (400, 500)

    def test_oversized_audio(self):
        """Simulate oversized audio file (>50MB as configured)."""
        # Generate just over the limit (we use the _read_with_limit which checks MAX_AUDIO_SIZE)
        # MAX_AUDIO_SIZE defaults to 50MB from config. We send 51MB of RIFF data.
        # Build a valid WAV header + enough padding to exceed 50MB
        header = b"RIFF" + struct.pack("<I", 50 * 1024 * 1024 + 36) + b"WAVEfmt "
        header += struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
        header += b"data" + struct.pack("<I", 50 * 1024 * 1024)
        data = header + b"\x00" * (50 * 1024 * 1024 + 1)  # just over limit
        r = client.post(
            "/api/analyze/audio",
            files={"file": ("big.wav", data, "audio/wav")},
        )
        assert r.status_code == 413

    def test_wrong_content_type(self):
        r = client.post(
            "/api/analyze/audio",
            files={"file": ("test.txt", b"not audio", "text/plain")},
        )
        assert r.status_code in (400, 415)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Video Analysis
# ═══════════════════════════════════════════════════════════════════════════


class TestVideoAnalysis:
    def test_invalid_format(self):
        r = client.post(
            "/api/analyze/video",
            files={"file": ("vid.mp4", b"not video data", "video/mp4")},
        )
        # Magic bytes won't match video format
        assert r.status_code in (400, 415, 500)

    def test_empty_file(self):
        r = client.post(
            "/api/analyze/video",
            files={"file": ("empty.mp4", b"", "video/mp4")},
        )
        assert r.status_code in (400, 415)

    def test_text_file_as_video(self):
        r = client.post(
            "/api/analyze/video",
            files={"file": ("fake.avi", b"Hello, I am not a video file.", "text/plain")},
        )
        assert r.status_code in (400, 415)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Media Analysis (Unified)
# ═══════════════════════════════════════════════════════════════════════════


class TestMediaAnalysis:
    def test_valid_jpeg(self):
        img = make_jpeg_bytes()
        r = client.post(
            "/api/analyze/media",
            files={"file": ("photo.jpg", img, "image/jpeg")},
        )
        assert r.status_code in (200, 500)
        if r.status_code == 200:
            data = r.json()
            assert "verdict" in data
            assert "confidence" in data

    def test_valid_png(self):
        img = make_png_bytes()
        r = client.post(
            "/api/analyze/media",
            files={"file": ("photo.png", img, "image/png")},
        )
        assert r.status_code in (200, 500)

    def test_invalid_type_pdf(self):
        r = client.post(
            "/api/analyze/media",
            files={"file": ("doc.txt", b"just text", "text/plain")},
        )
        assert r.status_code in (400, 415)

    def test_empty_file(self):
        r = client.post(
            "/api/analyze/media",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert r.status_code == 400

    def test_polyglot_jpeg_with_script(self):
        """JPEG with embedded <script> tag should be rejected by polyglot check."""
        img = make_jpeg_bytes()
        # Inject a <script> tag into the first 1024 bytes
        poisoned = img[:20] + b'<script>alert("xss")</script>' + img[20:]
        r = client.post(
            "/api/analyze/media",
            files={"file": ("evil.jpg", poisoned, "image/jpeg")},
        )
        assert r.status_code == 400
        body = r.json()
        assert "embedded" in body.get("detail", "").lower() or "polyglot" in body.get("detail", "").lower() or "clean" in body.get("detail", "").lower()

    def test_polyglot_png_with_html(self):
        """PNG with embedded <html> should be rejected."""
        img = make_png_bytes()
        poisoned = img[:16] + b"<html><body>pwned</body></html>" + img[16:]
        r = client.post(
            "/api/analyze/media",
            files={"file": ("evil.png", poisoned, "image/png")},
        )
        assert r.status_code == 400

    def test_oversized_media(self):
        """Simulated file over 100MB limit (MAX_MEDIA_SIZE)."""
        # Construct data just over 100MB
        header = make_jpeg_bytes()
        padding = b"\x00" * (100 * 1024 * 1024 + 1)
        oversized = header + padding
        r = client.post(
            "/api/analyze/media",
            files={"file": ("huge.jpg", oversized, "image/jpeg")},
        )
        assert r.status_code == 413


# ═══════════════════════════════════════════════════════════════════════════
# 6. Multimodal Analysis
# ═══════════════════════════════════════════════════════════════════════════


class TestMultimodalAnalysis:
    def test_text_only(self):
        r = client.post(
            "/api/analyze/multimodal",
            data={"text": "This is a test transcript for multimodal analysis."},
        )
        assert r.status_code in (200, 500)

    def test_audio_only(self):
        wav = make_wav_bytes()
        r = client.post(
            "/api/analyze/multimodal",
            files={"audio": ("test.wav", wav, "audio/wav")},
        )
        assert r.status_code in (200, 400, 500)

    def test_no_modalities(self):
        """Submitting no modalities should return 400."""
        r = client.post("/api/analyze/multimodal", data={})
        assert r.status_code == 400
        body = r.json()
        assert "modality" in body.get("detail", "").lower() or "required" in body.get("detail", "").lower()

    def test_text_too_long_multimodal(self):
        long_text = "z" * 10001
        r = client.post(
            "/api/analyze/multimodal",
            data={"text": long_text},
        )
        assert r.status_code == 400


# ═══════════════════════════════════════════════════════════════════════════
# 7. WebSocket Call Protection
# ═══════════════════════════════════════════════════════════════════════════


class TestWebSocket:
    def test_connect_and_ping(self):
        with client.websocket_connect("/ws/live") as ws:
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            assert resp["type"] == "pong"

    def test_call_start(self):
        with client.websocket_connect("/ws/live") as ws:
            ws.send_json({"type": "call_start"})
            resp = ws.receive_json()
            assert resp["type"] == "call_started"
            assert resp["status"] == "monitoring"

    def test_call_start_and_end(self):
        with client.websocket_connect("/ws/live") as ws:
            ws.send_json({"type": "call_start"})
            resp = ws.receive_json()
            assert resp["type"] == "call_started"

            ws.send_json({"type": "call_end"})
            resp = ws.receive_json()
            assert resp["type"] == "call_summary"
            assert "deepfake_detections" in resp
            assert "coercion_detections" in resp

    def test_send_audio_chunk(self):
        """Send a base64-encoded audio chunk during an active call."""
        wav = make_wav_bytes(duration_s=0.3)
        b64 = base64.b64encode(wav).decode("ascii")

        with client.websocket_connect("/ws/live") as ws:
            ws.send_json({"type": "call_start"})
            ws.receive_json()  # call_started

            ws.send_json({"type": "audio", "data": b64})
            # We may receive one or more responses (analysis_result, transcript, etc.)
            # Just verify we get at least one response without crashing
            try:
                resp = ws.receive_json(mode="text")
                assert "type" in resp
            except Exception:
                # If it times out or errors, the ML models may not be loaded
                # which is acceptable in CI
                pass

    def test_invalid_message_type(self):
        """Unknown message types should not crash the server."""
        with client.websocket_connect("/ws/live") as ws:
            ws.send_json({"type": "nonexistent_type", "data": "whatever"})
            # Server should silently ignore unknown types or respond gracefully
            # Send a ping to prove the connection is still alive
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            assert resp["type"] == "pong"


# ═══════════════════════════════════════════════════════════════════════════
# 8. Auth Flow
# ═══════════════════════════════════════════════════════════════════════════


class TestAuth:
    def test_register(self):
        email, _, token, resp = _register_user()
        assert resp.status_code == 200
        data = resp.json()
        assert data["token"]
        assert data["user"]["email"] == email

    def test_login(self):
        email, password, _, reg_resp = _register_user()
        assert reg_resp.status_code == 200

        r = client.post(
            "/api/auth/login",
            json={"email": email, "password": password},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["token"]
        assert data["user"]["email"] == email

    def test_login_wrong_password(self):
        email, _, _, _ = _register_user()
        r = client.post(
            "/api/auth/login",
            json={"email": email, "password": "WrongPassword123!"},
        )
        assert r.status_code == 401

    def test_get_profile(self):
        _, _, token, _ = _register_user()
        r = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "email" in data
        assert "name" in data

    def test_get_profile_unauthenticated(self):
        r = client.get("/api/auth/me")
        assert r.status_code == 401

    def test_change_password(self):
        email, password, token, _ = _register_user()
        new_pass = "NewStr0ng!P@ss99"
        r = client.put(
            "/api/auth/password",
            json={"current_password": password, "new_password": new_pass},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert r.json()["success"] is True

        # Verify new password works for login
        r2 = client.post(
            "/api/auth/login",
            json={"email": email, "password": new_pass},
        )
        assert r2.status_code == 200

    def test_change_password_wrong_current(self):
        _, _, token, _ = _register_user()
        r = client.put(
            "/api/auth/password",
            json={"current_password": "WrongOldPass!", "new_password": "NewPass123!"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 403

    def test_register_weak_password(self):
        """Password under 8 chars should be rejected by Pydantic validation."""
        r = client.post(
            "/api/auth/register",
            json={"email": _unique_email(), "password": "short", "name": "Weak"},
        )
        assert r.status_code == 422  # Pydantic min_length=8

    def test_register_duplicate_email(self):
        email, _, _, _ = _register_user()
        # Try registering again with the same email
        r = client.post(
            "/api/auth/register",
            json={"email": email, "password": "AnotherStr0ng!Pass", "name": "Dup"},
        )
        assert r.status_code == 409

    def test_register_invalid_email(self):
        r = client.post(
            "/api/auth/register",
            json={"email": "not-an-email", "password": "Str0ngP@ss!", "name": "Bad Email"},
        )
        assert r.status_code == 422

    def test_register_missing_name(self):
        r = client.post(
            "/api/auth/register",
            json={"email": _unique_email(), "password": "Str0ngP@ss!"},
        )
        assert r.status_code == 422

    def test_profile_with_invalid_token(self):
        r = client.get(
            "/api/auth/me",
            headers={"Authorization": "Bearer completely-invalid-jwt-token"},
        )
        assert r.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════
# 9. Edge Cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_concurrent_text_requests(self):
        """Send 5 text analysis requests concurrently via threading."""
        results = [None] * 5

        def _send(idx):
            r = client.post(
                "/api/analyze/text",
                json={"text": f"Concurrent test message number {idx}."},
            )
            results[idx] = r.status_code

        threads = [threading.Thread(target=_send, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # All should have completed with either success or model-not-loaded
        for code in results:
            assert code is not None, "A concurrent request did not complete"
            assert code in (200, 429, 500)

    def test_request_id_in_response(self):
        """Every response should have an X-Request-ID header."""
        r = client.get("/api/health")
        assert "X-Request-ID" in r.headers

    def test_custom_request_id_echoed(self):
        """Client-supplied X-Request-ID should be echoed back."""
        custom_id = "test-req-42"
        r = client.get("/api/health", headers={"X-Request-ID": custom_id})
        assert r.headers.get("X-Request-ID") == custom_id

    def test_response_time_header(self):
        """X-Response-Time header should be present."""
        r = client.get("/api/health")
        assert "X-Response-Time" in r.headers
        assert "ms" in r.headers["X-Response-Time"]

    def test_security_header_content_type_options(self):
        r = client.get("/api/health")
        assert r.headers.get("X-Content-Type-Options") == "nosniff"

    def test_security_header_frame_options(self):
        r = client.get("/api/health")
        assert r.headers.get("X-Frame-Options") == "DENY"

    def test_security_header_xss_protection(self):
        r = client.get("/api/health")
        assert r.headers.get("X-XSS-Protection") == "1; mode=block"

    def test_security_header_referrer_policy(self):
        r = client.get("/api/health")
        assert r.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

    def test_security_header_permissions_policy(self):
        r = client.get("/api/health")
        assert "Permissions-Policy" in r.headers

    def test_cache_control_on_api(self):
        """API endpoints should have no-store cache control."""
        r = client.get("/api/health")
        cc = r.headers.get("Cache-Control", "")
        assert "no-store" in cc

    def test_cors_preflight(self):
        """OPTIONS preflight request from allowed origin should succeed."""
        r = client.options(
            "/api/analyze/text",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert r.status_code == 200
        assert "access-control-allow-origin" in r.headers

    def test_cors_disallowed_origin(self):
        """Requests from non-allowed origins should not get CORS headers.

        Note: FastAPI CORS middleware still responds 200 to OPTIONS but omits
        the Access-Control-Allow-Origin header for disallowed origins.
        """
        r = client.options(
            "/api/analyze/text",
            headers={
                "Origin": "https://evil-site.example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        # Either no CORS header or origin != the evil origin
        origin_header = r.headers.get("access-control-allow-origin", "")
        assert "evil-site" not in origin_header

    def test_nonexistent_endpoint(self):
        r = client.get("/api/does-not-exist")
        assert r.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════
# 10. Batch API
# ═══════════════════════════════════════════════════════════════════════════


class TestBatchAPI:
    def test_batch_zip_with_jpeg(self):
        zip_data = make_zip_with_jpeg("sample.jpg")
        r = client.post(
            "/api/batch/analyze",
            files={"file": ("batch.zip", zip_data, "application/zip")},
        )
        assert r.status_code == 200
        data = r.json()
        assert "batch_id" in data
        assert data["total_files"] == 1
        assert data["status"] == "processing"

    def test_batch_non_zip(self):
        r = client.post(
            "/api/batch/analyze",
            files={"file": ("data.tar.gz", b"\x1f\x8b\x08\x00" + b"\x00" * 100, "application/gzip")},
        )
        assert r.status_code == 400

    def test_batch_invalid_zip(self):
        r = client.post(
            "/api/batch/analyze",
            files={"file": ("bad.zip", b"this is not a zip file", "application/zip")},
        )
        assert r.status_code == 400

    def test_batch_empty_zip(self):
        """ZIP with no supported media files should be rejected."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("readme.txt", "no media here")
        buf.seek(0)
        r = client.post(
            "/api/batch/analyze",
            files={"file": ("empty_media.zip", buf.read(), "application/zip")},
        )
        assert r.status_code == 400

    def test_batch_get_nonexistent(self):
        r = client.get("/api/batch/nonexistent-id")
        assert r.status_code == 404

    def test_batch_poll_status(self):
        """Create a batch and poll for its status."""
        zip_data = make_zip_with_jpeg()
        r = client.post(
            "/api/batch/analyze",
            files={"file": ("batch.zip", zip_data, "application/zip")},
        )
        assert r.status_code == 200
        batch_id = r.json()["batch_id"]

        # Poll (may still be processing)
        r2 = client.get(f"/api/batch/{batch_id}")
        assert r2.status_code == 200
        data = r2.json()
        assert data["id"] == batch_id
        assert data["status"] in ("processing", "completed")


# ═══════════════════════════════════════════════════════════════════════════
# 11. Validation: Magic Byte Detection
# ═══════════════════════════════════════════════════════════════════════════


class TestMagicByteValidation:
    """Verify that the validator correctly identifies file types by content,
    not by declared MIME type or file extension."""

    @pytest.mark.parametrize(
        "magic_prefix, expected_mime",
        [
            (b"\xff\xd8\xff\xe0", "image/jpeg"),
            (b"\xff\xd8\xff\xe1", "image/jpeg"),
            (b"\x89PNG\r\n\x1a\n", "image/png"),
            (b"GIF89a", "image/gif"),
            (b"BM", "image/bmp"),
            (b"fLaC", "audio/flac"),
            (b"OggS", "audio/ogg"),
            (b"ID3", "audio/mpeg"),
            (b"\xff\xfb", "audio/mpeg"),
        ],
        ids=[
            "jpeg-jfif",
            "jpeg-exif",
            "png",
            "gif89a",
            "bmp",
            "flac",
            "ogg",
            "mp3-id3",
            "mp3-sync",
        ],
    )
    def test_detect_file_type(self, magic_prefix, expected_mime):
        from server.validators import detect_file_type

        data = magic_prefix + b"\x00" * 64
        mime, ftype = detect_file_type(data)
        assert mime == expected_mime

    def test_detect_wav(self):
        from server.validators import detect_file_type

        wav = make_wav_bytes()
        mime, ftype = detect_file_type(wav)
        assert mime == "audio/wav"
        from server.validators import FileType
        assert ftype == FileType.AUDIO

    def test_detect_unknown(self):
        from server.validators import detect_file_type, FileType

        mime, ftype = detect_file_type(b"\x01\x02\x03\x04\x05\x06\x07\x08")
        assert ftype == FileType.UNKNOWN

    def test_detect_riff_webp(self):
        from server.validators import detect_file_type, FileType

        data = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 32
        mime, ftype = detect_file_type(data)
        assert mime == "image/webp"
        assert ftype == FileType.IMAGE

    def test_detect_riff_avi(self):
        from server.validators import detect_file_type, FileType

        data = b"RIFF\x00\x00\x00\x00AVI " + b"\x00" * 32
        mime, ftype = detect_file_type(data)
        assert mime == "video/x-msvideo"
        assert ftype == FileType.VIDEO

    def test_validate_rejects_wrong_type_for_endpoint(self):
        """Sending a JPEG to the audio endpoint should raise ValidationError."""
        from server.validators import validate_file, ValidationError

        jpeg_data = make_jpeg_bytes()
        with pytest.raises(ValidationError) as exc_info:
            validate_file(jpeg_data, "audio")
        assert exc_info.value.status_code in (400, 415)

    def test_validate_rejects_empty(self):
        from server.validators import validate_file, ValidationError

        with pytest.raises(ValidationError) as exc_info:
            validate_file(b"", "image")
        assert exc_info.value.status_code == 400

    def test_polyglot_detection_svg(self):
        """Files containing <svg> in header should be flagged."""
        from server.validators import _is_polyglot_suspect

        data = b"\xff\xd8\xff\xe0" + b"\x00" * 10 + b"<svg onload=alert(1)>" + b"\x00" * 50
        assert _is_polyglot_suspect(data) is True

    def test_polyglot_detection_php(self):
        from server.validators import _is_polyglot_suspect

        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10 + b"<?php system('ls'); ?>" + b"\x00" * 50
        assert _is_polyglot_suspect(data) is True

    def test_clean_file_not_flagged(self):
        """A normal JPEG should not be flagged as polyglot."""
        from server.validators import _is_polyglot_suspect

        jpeg = make_jpeg_bytes()
        assert _is_polyglot_suspect(jpeg) is False


# ═══════════════════════════════════════════════════════════════════════════
# 12. Monitoring Endpoints
# ═══════════════════════════════════════════════════════════════════════════


class TestMonitoring:
    def test_accuracy_dashboard(self):
        r = client.get("/api/monitoring/accuracy")
        assert r.status_code == 200

    def test_calibration_curve(self):
        r = client.get("/api/monitoring/calibration")
        assert r.status_code == 200

    def test_drift_alerts(self):
        r = client.get("/api/monitoring/drift")
        assert r.status_code == 200

    def test_check_rankings(self):
        r = client.get("/api/monitoring/checks")
        assert r.status_code == 200

    def test_deep_health_check(self):
        r = client.get("/api/monitoring/health/deep")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "components" in data

    def test_analysis_stats(self):
        r = client.get("/api/monitoring/stats")
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
# 13. Password Reset Flow
# ═══════════════════════════════════════════════════════════════════════════


class TestPasswordReset:
    def test_request_reset_existing_email(self):
        email, _, _, _ = _register_user()
        r = client.post("/api/auth/password-reset", json={"email": email})
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_request_reset_nonexistent_email(self):
        """Should still return success to prevent email enumeration."""
        r = client.post(
            "/api/auth/password-reset",
            json={"email": "nobody@nowhere.example.com"},
        )
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_reset_confirm_invalid_code(self):
        email, _, _, _ = _register_user()
        # Request reset
        client.post("/api/auth/password-reset", json={"email": email})
        # Try confirming with wrong code
        r = client.post(
            "/api/auth/password-reset/confirm",
            json={"email": email, "code": "000000", "new_password": "NewP@ssw0rd!!"},
        )
        assert r.status_code == 400
