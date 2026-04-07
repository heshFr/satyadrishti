"""
Satya Drishti — API Key Management Routes
==========================================
POST   /api/keys           — Create a new API key
GET    /api/keys           — List user's API keys
DELETE /api/keys/{key_id}  — Revoke an API key
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import User, ApiKey, generate_api_key
from ..auth import hash_password, verify_password, require_auth, get_current_user

log = logging.getLogger("satyadrishti.api_keys")
router = APIRouter(prefix="/api/keys", tags=["api-keys"])


class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    key_prefix: str
    is_active: bool
    last_used: str | None
    request_count: int
    created_at: str


class CreateKeyResponse(ApiKeyResponse):
    full_key: str  # Only returned on creation


@router.post("", response_model=CreateKeyResponse)
async def create_api_key(
    request: CreateKeyRequest,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    # Limit to 10 keys per user
    result = await db.execute(
        select(ApiKey).where(ApiKey.user_id == user.id, ApiKey.is_active == True)
    )
    active_keys = result.scalars().all()
    if len(active_keys) >= 10:
        raise HTTPException(status_code=400, detail="Maximum 10 active API keys allowed")

    raw_key = generate_api_key()
    key = ApiKey(
        user_id=user.id,
        name=request.name,
        key_hash=hash_password(raw_key),
        key_prefix=raw_key[:12] + "...",
    )
    db.add(key)
    await db.commit()
    await db.refresh(key)

    log.info("API key created: %s for user %s", key.key_prefix, user.email)
    return CreateKeyResponse(
        id=key.id,
        name=key.name,
        key_prefix=key.key_prefix,
        is_active=True,
        last_used=None,
        request_count=0,
        created_at=key.created_at.isoformat(),
        full_key=raw_key,
    )


@router.get("", response_model=list[ApiKeyResponse])
async def list_api_keys(
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ApiKey).where(ApiKey.user_id == user.id).order_by(ApiKey.created_at.desc())
    )
    keys = result.scalars().all()
    return [
        ApiKeyResponse(
            id=k.id,
            name=k.name,
            key_prefix=k.key_prefix,
            is_active=k.is_active,
            last_used=k.last_used.isoformat() if k.last_used else None,
            request_count=k.request_count or 0,
            created_at=k.created_at.isoformat(),
        )
        for k in keys
    ]


@router.delete("/{key_id}")
async def revoke_api_key(
    key_id: str,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == user.id)
    )
    key = result.scalar_one_or_none()
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")

    key.is_active = False
    await db.commit()
    log.info("API key revoked: %s for user %s", key.key_prefix, user.email)
    return {"status": "revoked"}


# ─── API Key Auth Middleware Helper ─────────────────────────────────────

async def authenticate_api_key(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """Check X-API-Key header and return the associated user, or None."""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return None

    # Search active keys
    result = await db.execute(select(ApiKey).where(ApiKey.is_active == True))
    keys = result.scalars().all()
    for k in keys:
        if verify_password(api_key, k.key_hash):
            # Update usage stats
            k.last_used = datetime.utcnow()
            k.request_count = (k.request_count or 0) + 1
            await db.commit()
            # Load user
            user_result = await db.execute(select(User).where(User.id == k.user_id))
            return user_result.scalar_one_or_none()

    return None
