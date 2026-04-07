"""
Satya Drishti — Remote Inference Client
========================================
HTTP client that forwards inference requests to a remote worker
(HuggingFace Spaces or Modal). Same interface as InferenceEngine
so the API gateway can swap between local and remote seamlessly.

Used when INFERENCE_URL environment variable is set.
"""

import io
import logging
import os
from typing import Dict, Any, Optional

import httpx

log = logging.getLogger("satyadrishti.inference_client")

# Generous timeouts: HF Spaces CPU inference can be slow
TIMEOUT_TEXT = 30.0
TIMEOUT_AUDIO = 120.0
TIMEOUT_MEDIA = 60.0
TIMEOUT_VIDEO = 300.0
TIMEOUT_MULTIMODAL = 360.0


class RemoteInferenceClient:
    """
    Drop-in replacement for InferenceEngine that forwards to a remote
    inference worker via HTTP. No torch/ML imports needed.
    """

    def __init__(self, base_url: str, token: str = ""):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._client: Optional[httpx.AsyncClient] = None
        log.info("Remote inference client targeting: %s", self.base_url)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {}
            if self.token:
                headers["X-Inference-Token"] = self.token
            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=httpx.Timeout(TIMEOUT_VIDEO, connect=15.0),
                follow_redirects=True,
            )
        return self._client

    async def _post_json(self, path: str, json_data: dict, timeout: float) -> Dict[str, Any]:
        client = self._get_client()
        try:
            resp = await client.post(
                f"{self.base_url}{path}",
                json=json_data,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException:
            return {"error": f"Inference timeout ({timeout}s). The service may be warming up — try again."}
        except httpx.HTTPStatusError as e:
            return {"error": f"Inference error: {e.response.status_code} {e.response.text[:200]}"}
        except Exception as e:
            return {"error": f"Inference unavailable: {e}"}

    async def _post_file(self, path: str, data: bytes, filename: str,
                         content_type: str, timeout: float,
                         extra_fields: Optional[dict] = None) -> Dict[str, Any]:
        client = self._get_client()
        try:
            files = {"file": (filename, data, content_type)}
            form_data = extra_fields or {}
            resp = await client.post(
                f"{self.base_url}{path}",
                files=files,
                data=form_data,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException:
            return {"error": f"Inference timeout ({timeout}s). The service may be warming up — try again."}
        except httpx.HTTPStatusError as e:
            return {"error": f"Inference error: {e.response.status_code} {e.response.text[:200]}"}
        except Exception as e:
            return {"error": f"Inference unavailable: {e}"}

    # ─── Public API (matches InferenceEngine interface) ───

    async def analyze_text(self, text: str) -> Dict[str, Any]:
        return await self._post_json("/infer/text", {"text": text}, TIMEOUT_TEXT)

    async def analyze_audio(self, audio_data: bytes) -> Dict[str, Any]:
        return await self._post_file(
            "/infer/audio", audio_data, "audio.wav", "audio/wav", TIMEOUT_AUDIO,
        )

    async def analyze_media(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, "rb") as f:
            data = f.read()
        ext = os.path.splitext(file_path)[1].lower()
        ct_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
        ct = ct_map.get(ext, "image/jpeg")
        return await self._post_file(
            "/infer/media", data, f"image{ext}", ct, TIMEOUT_MEDIA,
        )

    async def analyze_video(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, "rb") as f:
            data = f.read()
        ext = os.path.splitext(file_path)[1].lower()
        ct_map = {".mp4": "video/mp4", ".avi": "video/x-msvideo", ".mkv": "video/x-matroska", ".webm": "video/webm"}
        ct = ct_map.get(ext, "video/mp4")
        return await self._post_file(
            "/infer/video", data, f"video{ext}", ct, TIMEOUT_VIDEO,
        )

    async def analyze_multimodal(
        self,
        audio_data: bytes = None,
        video_path: str = None,
        text: str = None,
    ) -> Dict[str, Any]:
        client = self._get_client()
        try:
            files = {}
            form_data = {}

            if audio_data:
                files["audio"] = ("audio.wav", audio_data, "audio/wav")
            if video_path:
                with open(video_path, "rb") as f:
                    files["video"] = ("video.mp4", f.read(), "video/mp4")
            if text:
                form_data["text"] = text

            resp = await client.post(
                f"{self.base_url}/infer/multimodal",
                files=files or None,
                data=form_data or None,
                timeout=TIMEOUT_MULTIMODAL,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException:
            return {"error": f"Multimodal inference timeout ({TIMEOUT_MULTIMODAL}s)."}
        except Exception as e:
            return {"error": f"Multimodal inference unavailable: {e}"}

    async def transcribe_audio(self, audio_data: bytes, language: str = None) -> Dict[str, Any]:
        return await self._post_file(
            "/infer/transcribe", audio_data, "audio.wav", "audio/wav", TIMEOUT_AUDIO,
        )

    async def extract_audio_embedding(self, audio_data: bytes):
        """Not available remotely — return None (fusion falls back to rule-based)."""
        return None

    async def extract_text_embedding(self, text: str):
        return None

    async def extract_video_embedding(self, file_path: str):
        return None

    async def verify_speaker(self, audio_data: bytes) -> Dict[str, Any]:
        return {"is_verified": False, "error": "Speaker verification not available in remote mode"}

    async def enroll_voice_print(self, name: str, audio_data: bytes, relationship: str = "unknown") -> Dict[str, Any]:
        return {"status": "error", "message": "Voice enrollment not available in remote mode"}

    async def get_speaker_verifier(self):
        return None

    async def get_transcriber(self):
        return None

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
