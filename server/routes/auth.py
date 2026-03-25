"""
Satya Drishti — Auth Routes
===========================
POST /api/auth/register
POST /api/auth/login
GET  /api/auth/me
PUT  /api/auth/me
PUT  /api/auth/password
POST /api/auth/password-reset
POST /api/auth/password-reset/confirm
GET  /api/auth/oauth/{provider}/url
POST /api/auth/oauth/{provider}/callback
"""

import secrets
import logging
import random
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import User
from ..auth import hash_password, verify_password, create_access_token, require_auth
from ..config import (
    GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET,
    GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET,
    OAUTH_REDIRECT_URI,
)

log = logging.getLogger("satyadrishti.auth")
router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    name: str = Field(..., min_length=1, max_length=200)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


class UpdateProfileRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    language_pref: str | None = None


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=8, max_length=128)


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    code: str = Field(..., min_length=6, max_length=6)
    new_password: str = Field(..., min_length=8, max_length=128)


class UserResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    email: str
    name: str
    language_pref: str
    created_at: str


class TokenResponse(BaseModel):
    token: str
    user: UserResponse


@router.post("/register", response_model=TokenResponse)
async def register(request: RegisterRequest, db: AsyncSession = Depends(get_db)):
    # Check if email already exists
    result = await db.execute(select(User).where(User.email == request.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        email=request.email,
        password_hash=hash_password(request.password),
        name=request.name,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    token = create_access_token(user.id)
    return TokenResponse(
        token=token,
        user=UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            language_pref=user.language_pref,
            created_at=user.created_at.isoformat(),
        ),
    )


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(user.id)
    return TokenResponse(
        token=token,
        user=UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            language_pref=user.language_pref,
            created_at=user.created_at.isoformat(),
        ),
    )


@router.get("/me", response_model=UserResponse)
async def get_me(user: User = Depends(require_auth)):
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        language_pref=user.language_pref,
        created_at=user.created_at.isoformat(),
    )


@router.put("/me", response_model=UserResponse)
async def update_me(
    request: UpdateProfileRequest,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    if request.name is not None:
        user.name = request.name
    if request.language_pref is not None:
        if request.language_pref not in ("en", "hi", "mr", "hi-en"):
            raise HTTPException(status_code=400, detail="Supported languages: en, hi, mr, hi-en")
        user.language_pref = request.language_pref

    await db.commit()
    await db.refresh(user)

    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        language_pref=user.language_pref,
        created_at=user.created_at.isoformat(),
    )


@router.put("/password")
async def change_password(
    request: ChangePasswordRequest,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    if not verify_password(request.current_password, user.password_hash):
        raise HTTPException(status_code=403, detail="Current password is incorrect")

    user.password_hash = hash_password(request.new_password)
    await db.commit()
    log.info("Password changed for user %s", user.email)
    return {"success": True, "message": "Password updated successfully"}


# ═══════════════════════════════════════════════════════════════════════
# Password Reset (Forgot Password)
# ═══════════════════════════════════════════════════════════════════════

RESET_CODE_EXPIRY_MINUTES = 15


@router.post("/password-reset")
async def request_password_reset(
    request: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate a 6-digit reset code. Always returns success (prevents email enumeration)."""
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()

    if user:
        # Generate 6-digit code
        code = f"{random.randint(0, 999999):06d}"
        user.reset_token_hash = hash_password(code)
        user.reset_token_expires = datetime.utcnow() + timedelta(minutes=RESET_CODE_EXPIRY_MINUTES)
        await db.commit()
        log.info("Password reset code generated for %s: %s", user.email, code)
        # In production, send this via email. For dev, it's in server logs.
        return {
            "success": True,
            "message": f"If an account with that email exists, a reset code has been generated. Check your email or server logs.",
            "code": code,  # DEV ONLY — remove in production
        }

    # Same response shape even if user not found (prevents enumeration)
    return {
        "success": True,
        "message": "If an account with that email exists, a reset code has been generated. Check your email or server logs.",
    }


@router.post("/password-reset/confirm")
async def confirm_password_reset(
    request: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    """Validate the 6-digit code and set a new password."""
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()

    if not user or not user.reset_token_hash or not user.reset_token_expires:
        raise HTTPException(status_code=400, detail="Invalid or expired reset code")

    # Check expiry
    if datetime.utcnow() > user.reset_token_expires:
        # Clear expired token
        user.reset_token_hash = None
        user.reset_token_expires = None
        await db.commit()
        raise HTTPException(status_code=400, detail="Reset code has expired. Please request a new one.")

    # Verify code
    if not verify_password(request.code, user.reset_token_hash):
        raise HTTPException(status_code=400, detail="Invalid reset code")

    # Set new password and clear token
    user.password_hash = hash_password(request.new_password)
    user.reset_token_hash = None
    user.reset_token_expires = None
    await db.commit()

    log.info("Password reset completed for %s", user.email)
    return {"success": True, "message": "Password has been reset successfully"}


# ═══════════════════════════════════════════════════════════════════════
# OAuth — Google & GitHub
# ═══════════════════════════════════════════════════════════════════════

class OAuthCallbackRequest(BaseModel):
    code: str
    state: str | None = None


OAUTH_CONFIGS = {
    "google": {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "userinfo_url": "https://www.googleapis.com/oauth2/v3/userinfo",
        "scopes": "openid email profile",
    },
    "github": {
        "client_id": GITHUB_CLIENT_ID,
        "client_secret": GITHUB_CLIENT_SECRET,
        "auth_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "userinfo_url": "https://api.github.com/user",
        "scopes": "user:email read:user",
    },
}


@router.get("/oauth/{provider}/url")
async def get_oauth_url(provider: str):
    if provider not in OAUTH_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")

    cfg = OAUTH_CONFIGS[provider]
    if not cfg["client_id"]:
        raise HTTPException(status_code=501, detail=f"{provider} OAuth not configured on server")

    state = secrets.token_urlsafe(32)

    from urllib.parse import urlencode

    if provider == "google":
        params = urlencode({
            "client_id": cfg["client_id"],
            "redirect_uri": OAUTH_REDIRECT_URI,
            "response_type": "code",
            "scope": cfg["scopes"],
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        })
    else:  # github
        params = urlencode({
            "client_id": cfg["client_id"],
            "redirect_uri": OAUTH_REDIRECT_URI,
            "scope": cfg["scopes"],
            "state": state,
        })

    url = f"{cfg['auth_url']}?{params}"
    return {"url": url, "state": state}


@router.post("/oauth/{provider}/callback", response_model=TokenResponse)
async def oauth_callback(
    provider: str,
    request: OAuthCallbackRequest,
    db: AsyncSession = Depends(get_db),
):
    if provider not in OAUTH_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")

    cfg = OAUTH_CONFIGS[provider]
    if not cfg["client_id"] or not cfg["client_secret"]:
        raise HTTPException(status_code=501, detail=f"{provider} OAuth not configured on server")

    import httpx

    # Exchange code for access token
    async with httpx.AsyncClient() as client:
        token_data = {
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "code": request.code,
            "redirect_uri": OAUTH_REDIRECT_URI,
        }

        if provider == "google":
            token_data["grant_type"] = "authorization_code"
            token_resp = await client.post(cfg["token_url"], data=token_data)
        else:  # github
            token_resp = await client.post(
                cfg["token_url"], data=token_data,
                headers={"Accept": "application/json"},
            )

        if token_resp.status_code != 200:
            log.error("OAuth token exchange failed: %s", token_resp.text)
            raise HTTPException(status_code=400, detail="OAuth token exchange failed")

        token_json = token_resp.json()
        access_token = token_json.get("access_token")
        if not access_token:
            raise HTTPException(status_code=400, detail="No access token received")

        # Fetch user info
        headers = {"Authorization": f"Bearer {access_token}"}
        if provider == "github":
            headers["Accept"] = "application/json"

        user_resp = await client.get(cfg["userinfo_url"], headers=headers)
        if user_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch user profile")

        user_info = user_resp.json()

    # Extract email and name
    if provider == "google":
        email = user_info.get("email")
        name = user_info.get("name", email.split("@")[0] if email else "User")
    else:  # github
        email = user_info.get("email")
        name = user_info.get("name") or user_info.get("login", "User")
        # GitHub may not return email — fetch from emails endpoint
        if not email:
            async with httpx.AsyncClient() as client:
                emails_resp = await client.get(
                    "https://api.github.com/user/emails",
                    headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
                )
                if emails_resp.status_code == 200:
                    for e in emails_resp.json():
                        if e.get("primary") and e.get("verified"):
                            email = e["email"]
                            break

    if not email:
        raise HTTPException(status_code=400, detail="Could not retrieve email from OAuth provider")

    # Find or create user
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if not user:
        user = User(
            email=email,
            password_hash=hash_password(secrets.token_urlsafe(32)),
            name=name,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        log.info("OAuth user created: %s via %s", email, provider)
    else:
        log.info("OAuth login: %s via %s", email, provider)

    token = create_access_token(user.id)
    return TokenResponse(
        token=token,
        user=UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            language_pref=user.language_pref,
            created_at=user.created_at.isoformat(),
        ),
    )
