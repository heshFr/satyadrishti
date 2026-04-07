"""
Input Validation & Sanitization
=================================
Centralized validation for all file uploads and API inputs.
Prevents malformed files, adversarial inputs, and DoS attacks.

Security Layers:
1. Magic byte validation (file type by content, not extension)
2. Size limit enforcement
3. Timeout protection on analysis operations
4. Structured error responses
5. Content safety checks (no embedded scripts, no polyglot files)
"""

import asyncio
import io
import logging
import struct
from enum import Enum
from typing import Dict, Any, Optional, Tuple

log = logging.getLogger("satyadrishti.validators")


# ── File Type Definitions ──
# Magic bytes for supported media types

MAGIC_BYTES = {
    # Images
    "jpeg": [
        (b"\xff\xd8\xff\xe0", "image/jpeg"),  # JFIF
        (b"\xff\xd8\xff\xe1", "image/jpeg"),  # EXIF
        (b"\xff\xd8\xff\xee", "image/jpeg"),  # Adobe
        (b"\xff\xd8\xff\xdb", "image/jpeg"),  # Raw JPEG
        (b"\xff\xd8\xff\xe2", "image/jpeg"),  # ICC Profile
        (b"\xff\xd8\xff\xe8", "image/jpeg"),  # SPIFF
    ],
    "png": [
        (b"\x89PNG\r\n\x1a\n", "image/png"),
    ],
    "gif": [
        (b"GIF87a", "image/gif"),
        (b"GIF89a", "image/gif"),
    ],
    "webp": [
        (b"RIFF", "image/webp"),  # check "WEBP" at offset 8
    ],
    "bmp": [
        (b"BM", "image/bmp"),
    ],
    "tiff": [
        (b"II\x2a\x00", "image/tiff"),  # little-endian
        (b"MM\x00\x2a", "image/tiff"),  # big-endian
    ],
    "heif": [
        (b"\x00\x00\x00", "image/heif"),  # needs ftyp check at offset 4
    ],

    # Audio
    "wav": [
        (b"RIFF", "audio/wav"),  # check "WAVE" at offset 8
    ],
    "mp3": [
        (b"\xff\xfb", "audio/mpeg"),
        (b"\xff\xf3", "audio/mpeg"),
        (b"\xff\xf2", "audio/mpeg"),
        (b"ID3", "audio/mpeg"),  # ID3 tag
    ],
    "flac": [
        (b"fLaC", "audio/flac"),
    ],
    "ogg": [
        (b"OggS", "audio/ogg"),
    ],
    "m4a": [
        (b"\x00\x00\x00", "audio/mp4"),  # needs ftyp check
    ],
    "webm_audio": [
        (b"\x1a\x45\xdf\xa3", "audio/webm"),  # EBML header (Matroska/WebM)
    ],

    # Video
    "mp4": [
        (b"\x00\x00\x00", "video/mp4"),  # needs ftyp check at offset 4
    ],
    "avi": [
        (b"RIFF", "video/x-msvideo"),  # check "AVI " at offset 8
    ],
    "mkv": [
        (b"\x1a\x45\xdf\xa3", "video/x-matroska"),
    ],
    "webm": [
        (b"\x1a\x45\xdf\xa3", "video/webm"),
    ],
    "mov": [
        (b"\x00\x00\x00", "video/quicktime"),  # needs ftyp check
    ],
}

# Allowed MIME types per endpoint
ALLOWED_TYPES = {
    "image": {
        "image/jpeg", "image/png", "image/gif", "image/webp",
        "image/bmp", "image/tiff", "image/heif",
    },
    "audio": {
        "audio/wav", "audio/mpeg", "audio/flac", "audio/ogg",
        "audio/mp4", "audio/webm", "audio/x-wav", "audio/wave",
    },
    "video": {
        "video/mp4", "video/x-msvideo", "video/x-matroska",
        "video/webm", "video/quicktime", "video/mpeg",
    },
    "media": {
        "image/jpeg", "image/png", "image/gif", "image/webp",
        "image/bmp", "image/tiff", "image/heif",
        "video/mp4", "video/x-msvideo", "video/x-matroska",
        "video/webm", "video/quicktime", "video/mpeg",
    },
}

# Size limits (bytes)
SIZE_LIMITS = {
    "image": 50 * 1024 * 1024,       # 50 MB
    "audio": 100 * 1024 * 1024,      # 100 MB
    "video": 500 * 1024 * 1024,      # 500 MB
    "text": 1 * 1024 * 1024,         # 1 MB
}

# Analysis timeout (seconds)
ANALYSIS_TIMEOUTS = {
    "image": 30,
    "audio": 60,
    "video": 180,
    "text": 15,
    "multimodal": 240,
}


class FileType(Enum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"
    UNKNOWN = "unknown"


class ValidationError(Exception):
    """Raised when file validation fails."""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def detect_file_type(data: bytes) -> Tuple[str, FileType]:
    """
    Detect file type from magic bytes.

    Args:
        data: First 32+ bytes of the file

    Returns:
        (mime_type, file_type_category)
    """
    if len(data) < 4:
        return "application/octet-stream", FileType.UNKNOWN

    header = data[:16]

    # Check RIFF container (WAV, AVI, WEBP)
    if header[:4] == b"RIFF" and len(data) >= 12:
        subtype = data[8:12]
        if subtype == b"WAVE":
            return "audio/wav", FileType.AUDIO
        elif subtype == b"AVI ":
            return "video/x-msvideo", FileType.VIDEO
        elif subtype == b"WEBP":
            return "image/webp", FileType.IMAGE

    # Check ISO Base Media File Format (MP4, MOV, M4A, HEIF)
    if len(data) >= 12:
        ftyp_offset = data[4:8]
        if ftyp_offset == b"ftyp":
            brand = data[8:12].decode("ascii", errors="ignore").strip("\x00")
            video_brands = {"isom", "iso2", "mp41", "mp42", "avc1", "dash", "M4V "}
            audio_brands = {"M4A ", "mp4a", "M4B "}
            heif_brands = {"heic", "heix", "hevc", "mif1"}
            mov_brands = {"qt  "}

            if brand in heif_brands:
                return "image/heif", FileType.IMAGE
            elif brand in audio_brands:
                return "audio/mp4", FileType.AUDIO
            elif brand in mov_brands:
                return "video/quicktime", FileType.VIDEO
            elif brand in video_brands or brand.startswith("mp4"):
                return "video/mp4", FileType.VIDEO
            else:
                return "video/mp4", FileType.VIDEO

    # EBML (Matroska/WebM)
    if header[:4] == b"\x1a\x45\xdf\xa3":
        # Could be MKV or WebM — default to video
        return "video/webm", FileType.VIDEO

    # JPEG
    if header[:2] == b"\xff\xd8":
        return "image/jpeg", FileType.IMAGE

    # PNG
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png", FileType.IMAGE

    # GIF
    if header[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif", FileType.IMAGE

    # BMP
    if header[:2] == b"BM":
        return "image/bmp", FileType.IMAGE

    # TIFF
    if header[:4] in (b"II\x2a\x00", b"MM\x00\x2a"):
        return "image/tiff", FileType.IMAGE

    # FLAC
    if header[:4] == b"fLaC":
        return "audio/flac", FileType.AUDIO

    # OGG
    if header[:4] == b"OggS":
        return "audio/ogg", FileType.AUDIO

    # MP3
    if header[:3] == b"ID3" or header[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg", FileType.AUDIO

    return "application/octet-stream", FileType.UNKNOWN


def validate_file(
    data: bytes,
    endpoint: str,
    declared_mime: Optional[str] = None,
) -> Tuple[str, FileType]:
    """
    Validate uploaded file against expected type for endpoint.

    Args:
        data: File content (bytes)
        endpoint: API endpoint name ("image", "audio", "video", "media")
        declared_mime: MIME type declared by the client (optional)

    Returns:
        (actual_mime_type, file_type_category)

    Raises:
        ValidationError: If file is invalid or disallowed type
    """
    if len(data) == 0:
        raise ValidationError("Empty file uploaded", 400)

    # Size check
    max_size = SIZE_LIMITS.get(endpoint, SIZE_LIMITS.get("video"))
    if len(data) > max_size:
        max_mb = max_size / (1024 * 1024)
        actual_mb = len(data) / (1024 * 1024)
        raise ValidationError(
            f"File too large: {actual_mb:.1f}MB exceeds {max_mb:.0f}MB limit for {endpoint}",
            413,
        )

    # Detect actual type from magic bytes
    actual_mime, file_type = detect_file_type(data)

    if file_type == FileType.UNKNOWN:
        raise ValidationError(
            f"Unrecognized file type. Supported: images (JPEG/PNG/WebP), "
            f"audio (WAV/MP3/FLAC/OGG), video (MP4/AVI/MKV/WebM)",
            415,
        )

    # Check against allowed types for this endpoint
    allowed = ALLOWED_TYPES.get(endpoint, ALLOWED_TYPES.get("media", set()))
    if actual_mime not in allowed:
        raise ValidationError(
            f"File type '{actual_mime}' not allowed for /{endpoint} endpoint. "
            f"Expected: {', '.join(sorted(allowed))}",
            415,
        )

    # Cross-check with declared MIME (warn but don't reject)
    if declared_mime and not _mime_compatible(declared_mime, actual_mime):
        log.warning(
            "MIME mismatch: client declared '%s' but file content is '%s'",
            declared_mime, actual_mime,
        )

    # Polyglot check: reject files that look like they might be crafted
    # to be interpreted differently by different parsers
    if _is_polyglot_suspect(data):
        raise ValidationError(
            "File appears to contain embedded content of a different type. "
            "Please upload a clean media file.",
            400,
        )

    return actual_mime, file_type


def _mime_compatible(declared: str, actual: str) -> bool:
    """Check if declared and actual MIME types are compatible."""
    # Same type
    if declared == actual:
        return True

    # Same category (e.g., audio/mpeg vs audio/mp3)
    d_cat = declared.split("/")[0]
    a_cat = actual.split("/")[0]
    if d_cat == a_cat:
        return True

    # WAV variants
    wav_types = {"audio/wav", "audio/wave", "audio/x-wav"}
    if declared in wav_types and actual in wav_types:
        return True

    # Video/audio overlap for containers (MP4 can be either)
    mp4_types = {"video/mp4", "audio/mp4", "video/quicktime"}
    if declared in mp4_types and actual in mp4_types:
        return True

    return False


def _is_polyglot_suspect(data: bytes) -> bool:
    """
    Basic check for polyglot files (files crafted to be valid in
    multiple formats simultaneously).
    """
    # Check for HTML/script content in image/audio headers
    # This catches basic polyglot attacks
    header = data[:1024].lower()

    suspicious_patterns = [
        b"<script",
        b"<html",
        b"<svg",
        b"javascript:",
        b"data:text/html",
        b"<?php",
    ]

    for pattern in suspicious_patterns:
        if pattern in header:
            return True

    return False


async def with_timeout(
    coro,
    endpoint: str,
    timeout_override: Optional[int] = None,
):
    """
    Run an async coroutine with a timeout appropriate for the endpoint.

    Args:
        coro: Async coroutine to run
        endpoint: Endpoint name for timeout lookup
        timeout_override: Optional custom timeout in seconds

    Returns:
        Coroutine result

    Raises:
        ValidationError: If timeout is exceeded
    """
    timeout = timeout_override or ANALYSIS_TIMEOUTS.get(endpoint, 60)

    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise ValidationError(
            f"Analysis timed out after {timeout}s. "
            f"The file may be too large or complex. Try a shorter/smaller file.",
            504,
        )


def validate_text_input(
    text: str,
    max_length: int = 10000,
    min_length: int = 1,
) -> str:
    """
    Validate text input for coercion analysis.

    Args:
        text: Input text
        max_length: Maximum allowed characters
        min_length: Minimum required characters

    Returns:
        Sanitized text

    Raises:
        ValidationError: If text is invalid
    """
    if not text or len(text.strip()) < min_length:
        raise ValidationError(
            f"Text too short. Minimum {min_length} characters required.",
            400,
        )

    if len(text) > max_length:
        raise ValidationError(
            f"Text too long: {len(text)} characters exceeds {max_length} limit.",
            400,
        )

    # Basic sanitization (strip null bytes, normalize whitespace)
    sanitized = text.replace("\x00", "").strip()

    return sanitized


def get_analysis_timeout(endpoint: str) -> int:
    """Get timeout in seconds for a given endpoint."""
    return ANALYSIS_TIMEOUTS.get(endpoint, 60)
