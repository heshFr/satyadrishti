"""
Satya Drishti — Audit Archive
==============================
Covert archival of uploaded media for security auditing.
Files are saved before temp cleanup, organized by date and scan ID.
This module is NOT exposed to users — the privacy policy states files are deleted.
"""

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from .config import PROJECT_ROOT

log = logging.getLogger("satyadrishti.audit")

ARCHIVE_DIR = PROJECT_ROOT / "data" / ".audit_archive"

# Sub-directories per modality
_MODALITY_DIRS = {
    "image": "images",
    "video": "videos",
    "audio": "audio",
    "unknown": "other",
}


def _ensure_dirs():
    """Create archive directory structure if missing."""
    for subdir in _MODALITY_DIRS.values():
        (ARCHIVE_DIR / subdir).mkdir(parents=True, exist_ok=True)


def archive_file(
    source_path: str,
    modality: str,
    original_filename: str | None = None,
    scan_id: str | None = None,
    user_id: str | None = None,
) -> str | None:
    """
    Copy a file into the audit archive. Returns the archive path or None on failure.

    Directory structure:
        data/.audit_archive/{modality}/{YYYY-MM-DD}/{timestamp}_{scan_id}_{original_name}.ext
    """
    try:
        _ensure_dirs()
        subdir = _MODALITY_DIRS.get(modality, "other")
        date_dir = ARCHIVE_DIR / subdir / datetime.utcnow().strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)

        # Build filename
        ts = datetime.utcnow().strftime("%H%M%S")
        ext = Path(source_path).suffix or ".bin"
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in (original_filename or "upload"))
        sid = (scan_id or "anon")[:12]
        uid = (user_id or "anon")[:8]
        dest_name = f"{ts}_{uid}_{sid}_{safe_name}{ext}" if not safe_name.endswith(ext) else f"{ts}_{uid}_{sid}_{safe_name}"
        dest_path = date_dir / dest_name

        shutil.copy2(source_path, dest_path)
        log.debug("Archived %s → %s", source_path, dest_path)
        return str(dest_path)
    except Exception as e:
        log.warning("Audit archive failed for %s: %s", source_path, e)
        return None


def archive_bytes(
    data: bytes,
    modality: str,
    extension: str = ".bin",
    original_filename: str | None = None,
    scan_id: str | None = None,
    user_id: str | None = None,
) -> str | None:
    """Archive raw bytes (e.g. audio uploaded as bytes, not temp file)."""
    try:
        _ensure_dirs()
        subdir = _MODALITY_DIRS.get(modality, "other")
        date_dir = ARCHIVE_DIR / subdir / datetime.utcnow().strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.utcnow().strftime("%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in (original_filename or "upload"))
        sid = (scan_id or "anon")[:12]
        uid = (user_id or "anon")[:8]
        dest_name = f"{ts}_{uid}_{sid}_{safe_name}{extension}"
        dest_path = date_dir / dest_name

        dest_path.write_bytes(data)
        log.debug("Archived bytes → %s (%d bytes)", dest_path, len(data))
        return str(dest_path)
    except Exception as e:
        log.warning("Audit archive bytes failed: %s", e)
        return None


def delete_user_archives(user_id: str) -> int:
    """Delete all archived files for a user (right to erasure). Returns count deleted."""
    if not ARCHIVE_DIR.exists():
        return 0
    count = 0
    uid_prefix = user_id[:8]
    for root, _, files in os.walk(ARCHIVE_DIR):
        for f in files:
            if f"_{uid_prefix}_" in f:
                try:
                    os.remove(os.path.join(root, f))
                    count += 1
                except OSError:
                    pass
    return count
