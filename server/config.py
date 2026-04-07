"""
Satya Drishti — Configuration
==============================
Centralized settings loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path

# ─── Paths ───
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DB_PATH = PROJECT_ROOT / "satyadrishti.db"

# ─── Database ───
_raw_db_url = os.environ.get("SATYA_DATABASE_URL") or os.environ.get("DATABASE_URL") or ""
if _raw_db_url:
    # Render/Railway provide postgres:// but SQLAlchemy needs postgresql+asyncpg://
    if _raw_db_url.startswith("postgres://"):
        _raw_db_url = _raw_db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif _raw_db_url.startswith("postgresql://"):
        _raw_db_url = _raw_db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    # Neon/psycopg2 use ?sslmode=require, but asyncpg uses ?ssl=require.
    # Rewrite the query parameter so the same DSN works for both drivers.
    if "sslmode=" in _raw_db_url:
        _raw_db_url = _raw_db_url.replace("sslmode=require", "ssl=require")
        _raw_db_url = _raw_db_url.replace("sslmode=verify-full", "ssl=verify-full")
        _raw_db_url = _raw_db_url.replace("sslmode=prefer", "ssl=prefer")
        _raw_db_url = _raw_db_url.replace("sslmode=disable", "ssl=disable")
    DATABASE_URL = _raw_db_url
else:
    DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

# ─── Auth ───
JWT_SECRET = os.environ.get("SATYA_JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = int(os.environ.get("SATYA_JWT_EXPIRY_HOURS", "24"))

# ─── OAuth ───
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GITHUB_CLIENT_ID = os.environ.get("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.environ.get("GITHUB_CLIENT_SECRET", "")
OAUTH_REDIRECT_URI = os.environ.get("OAUTH_REDIRECT_URI", "http://localhost:3000/auth/callback")

# ─── CORS ───
CORS_ORIGINS = os.environ.get(
    "SATYA_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:3001,http://localhost:5173,https://satyadrishti.vercel.app",
).split(",")

# ─── Server ───
HOST = os.environ.get("SATYA_HOST", "0.0.0.0")
PORT = int(os.environ.get("SATYA_PORT", "8000"))

# ─── Remote Inference (Zero-Cost Architecture) ───
# When set, the API gateway forwards ML inference to a remote worker
# (e.g. HuggingFace Spaces) instead of loading models locally.
# This keeps the gateway under 512MB RAM for free-tier hosting.
INFERENCE_URL = os.environ.get("INFERENCE_URL", "")  # e.g. "https://user-satyadrishti.hf.space"
INFERENCE_SECRET = os.environ.get("INFERENCE_SECRET", "")

# ─── Email (SMTP) ───
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("SMTP_FROM", "noreply@satyadrishti.in")
SMTP_TLS = os.environ.get("SMTP_TLS", "true").lower() == "true"

# ─── Frontend URL (for email links) ───
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000")

# ─── Upload Limits (bytes) ───
MAX_AUDIO_SIZE = int(os.environ.get("SATYA_MAX_AUDIO_MB", "50")) * 1024 * 1024
MAX_VIDEO_SIZE = int(os.environ.get("SATYA_MAX_VIDEO_MB", "200")) * 1024 * 1024
MAX_MEDIA_SIZE = int(os.environ.get("SATYA_MAX_MEDIA_MB", "100")) * 1024 * 1024
MAX_TEXT_LENGTH = int(os.environ.get("SATYA_MAX_TEXT_LENGTH", "10000"))

# ─── Model Checkpoints ───
AUDIO_MODEL = None  # HuggingFace pretrained, no local checkpoint
XLS_R_MODEL_PATH = str(MODEL_DIR / "audio" / "xls_r_300m")
TEXT_CHECKPOINT = str(MODEL_DIR / "text" / "deberta_coercion_lora" / "best_model")
VIDEO_SPATIAL_CKPT = str(MODEL_DIR / "video" / "vit_spatial_v2_best.pt")
VIDEO_TEMPORAL_CKPT = str(MODEL_DIR / "video" / "r3d_temporal_v2_best.pt")
FORENSICS_CKPT = str(MODEL_DIR / "image_forensics" / "deepfake_vit_b16.pt")
FORENSICS_PRETRAINED = str(MODEL_DIR / "image_forensics" / "pretrained_vit")
FUSION_CKPT = str(MODEL_DIR / "fusion" / "fusion_network_best.pt")


def validate_jwt_secret():
    """Warn if using empty/default JWT secret."""
    if not JWT_SECRET:
        import warnings
        warnings.warn(
            "SATYA_JWT_SECRET not set! Using insecure default. "
            "Set this environment variable in production.",
            stacklevel=2,
        )
        return "satyadrishti-dev-insecure-default-key"
    return JWT_SECRET
