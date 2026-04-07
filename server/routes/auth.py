"""
Satya Drishti — Auth Routes
===========================
POST /api/auth/register      — Create account + send verification email
POST /api/auth/login          — Email/password login (2FA-aware)
GET  /api/auth/me             — Current user profile
PUT  /api/auth/me             — Update profile + notification prefs
PUT  /api/auth/password       — Change password
POST /api/auth/password-reset — Request password reset code
POST /api/auth/password-reset/confirm — Confirm reset code
POST /api/auth/verify-email   — Verify email with code
POST /api/auth/resend-verification — Resend verification email
POST /api/auth/2fa/setup      — Generate TOTP secret + QR URL
POST /api/auth/2fa/confirm    — Confirm 2FA with first TOTP code
POST /api/auth/2fa/verify     — Verify 2FA during login
POST /api/auth/2fa/disable    — Disable 2FA
GET  /api/auth/oauth/{provider}/url
POST /api/auth/oauth/{provider}/callback
"""

import secrets
import logging
import hashlib
import base64
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import User, Scan, Case, ApiKey
from ..auth import hash_password, verify_password, create_access_token, require_auth
from ..config import (
    GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET,
    GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET,
    OAUTH_REDIRECT_URI,
)
from ..email_service import send_verification_email, send_password_reset_email
from ..rate_limiter import limiter

# Lazy-import pyotp (optional dependency)
try:
    import pyotp
    HAS_TOTP = True
except ImportError:
    HAS_TOTP = False

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
    notify_email_threats: bool | None = None
    notify_email_reports: bool | None = None
    notify_push_enabled: bool | None = None
    emergency_contact_name: str | None = Field(None, max_length=200)
    emergency_contact_phone: str | None = Field(None, max_length=20)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=8, max_length=128)


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    code: str = Field(..., min_length=6, max_length=6)
    new_password: str = Field(..., min_length=8, max_length=128)


class VerifyEmailRequest(BaseModel):
    code: str = Field(..., min_length=6, max_length=6)


class TwoFactorCodeRequest(BaseModel):
    code: str = Field(..., min_length=6, max_length=6)


class TwoFactorVerifyRequest(BaseModel):
    temp_token: str
    code: str = Field(..., min_length=6, max_length=8)  # 6 for TOTP, 8 for backup


class UserResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    email: str
    name: str
    language_pref: str
    email_verified: bool
    totp_enabled: bool
    oauth_provider: str | None
    notify_email_threats: bool
    notify_email_reports: bool
    notify_push_enabled: bool
    emergency_contact_name: str | None
    emergency_contact_phone: str | None
    created_at: str


class TokenResponse(BaseModel):
    token: str
    user: UserResponse
    requires_2fa: bool = False
    temp_token: str | None = None


def _user_resp(user: User) -> UserResponse:
    """Build a UserResponse from a User ORM object."""
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        language_pref=user.language_pref,
        email_verified=user.email_verified or False,
        totp_enabled=user.totp_enabled or False,
        oauth_provider=user.oauth_provider,
        notify_email_threats=user.notify_email_threats if user.notify_email_threats is not None else True,
        notify_email_reports=user.notify_email_reports if user.notify_email_reports is not None else False,
        notify_push_enabled=user.notify_push_enabled if user.notify_push_enabled is not None else False,
        emergency_contact_name=user.emergency_contact_name,
        emergency_contact_phone=user.emergency_contact_phone,
        created_at=user.created_at.isoformat(),
    )


@router.post("/register", response_model=TokenResponse)
async def register(request: RegisterRequest, req: Request = None, db: AsyncSession = Depends(get_db)):
    if req:
        limiter.check(req, limit=5, window=60, endpoint="auth_register")
    result = await db.execute(select(User).where(User.email == request.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    # Generate email verification code (cryptographically secure)
    verify_code = f"{secrets.randbelow(1000000):06d}"

    user = User(
        email=request.email,
        password_hash=hash_password(request.password),
        name=request.name,
        email_verify_code=hash_password(verify_code),
        email_verify_expires=datetime.utcnow() + timedelta(hours=24),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    # Send verification email (async-safe, won't block if SMTP unavailable)
    sent = send_verification_email(user.email, user.name, verify_code)
    if not sent:
        log.info("Email verification code for %s: %s (SMTP not configured)", user.email, verify_code)

    token = create_access_token(user.id)
    return TokenResponse(token=token, user=_user_resp(user))


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, req: Request = None, db: AsyncSession = Depends(get_db)):
    if req:
        limiter.check(req, limit=10, window=60, endpoint="auth_login")
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # If 2FA is enabled, return a temporary token instead of a full session
    if user.totp_enabled:
        temp_token = secrets.token_urlsafe(32)
        # Store temp token in reset_token_hash (reuse column, short-lived)
        user.reset_token_hash = hash_password(temp_token)
        user.reset_token_expires = datetime.utcnow() + timedelta(minutes=5)
        await db.commit()
        return TokenResponse(
            token="",
            user=_user_resp(user),
            requires_2fa=True,
            temp_token=temp_token,
        )

    token = create_access_token(user.id)
    return TokenResponse(token=token, user=_user_resp(user))


@router.get("/me", response_model=UserResponse)
async def get_me(user: User = Depends(require_auth)):
    return _user_resp(user)


@router.put("/me", response_model=UserResponse)
async def update_me(
    request: UpdateProfileRequest,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    SUPPORTED_LANGS = ("en", "hi", "mr", "ta", "te", "bn", "hi-en")
    if request.name is not None:
        user.name = request.name
    if request.language_pref is not None:
        if request.language_pref not in SUPPORTED_LANGS:
            raise HTTPException(status_code=400, detail=f"Supported languages: {', '.join(SUPPORTED_LANGS)}")
        user.language_pref = request.language_pref
    if request.notify_email_threats is not None:
        user.notify_email_threats = request.notify_email_threats
    if request.notify_email_reports is not None:
        user.notify_email_reports = request.notify_email_reports
    if request.notify_push_enabled is not None:
        user.notify_push_enabled = request.notify_push_enabled
    if request.emergency_contact_name is not None:
        user.emergency_contact_name = request.emergency_contact_name
    if request.emergency_contact_phone is not None:
        user.emergency_contact_phone = request.emergency_contact_phone

    await db.commit()
    await db.refresh(user)
    return _user_resp(user)


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
    req: Request = None,
    db: AsyncSession = Depends(get_db),
):
    """Generate a 6-digit reset code. Always returns success (prevents email enumeration)."""
    if req:
        limiter.check(req, limit=3, window=300, endpoint="password_reset")
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()

    if user:
        code = f"{secrets.randbelow(1000000):06d}"
        user.reset_token_hash = hash_password(code)
        user.reset_token_expires = datetime.utcnow() + timedelta(minutes=RESET_CODE_EXPIRY_MINUTES)
        await db.commit()
        sent = send_password_reset_email(user.email, user.name, code)
        if not sent:
            log.info("Password reset code for %s: %s (SMTP not configured)", user.email, code)
        return {
            "success": True,
            "message": "If an account with that email exists, a reset code has been sent.",
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
            oauth_provider=provider,
            email_verified=True,  # OAuth emails are already verified by the provider
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        log.info("OAuth user created: %s via %s", email, provider)
    else:
        if not user.oauth_provider:
            user.oauth_provider = provider
            await db.commit()
        log.info("OAuth login: %s via %s", email, provider)

    token = create_access_token(user.id)
    return TokenResponse(token=token, user=_user_resp(user))


# ═══════════════════════════════════════════════════════════════════════
# Email Verification
# ═══════════════════════════════════════════════════════════════════════

@router.post("/verify-email")
async def verify_email(
    request: VerifyEmailRequest,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    if user.email_verified:
        return {"success": True, "message": "Email already verified"}
    if not user.email_verify_code or not user.email_verify_expires:
        raise HTTPException(status_code=400, detail="No verification code pending. Request a new one.")
    if datetime.utcnow() > user.email_verify_expires:
        raise HTTPException(status_code=400, detail="Verification code expired. Request a new one.")
    if not verify_password(request.code, user.email_verify_code):
        raise HTTPException(status_code=400, detail="Invalid verification code")

    user.email_verified = True
    user.email_verify_code = None
    user.email_verify_expires = None
    await db.commit()
    log.info("Email verified for %s", user.email)
    return {"success": True, "message": "Email verified successfully"}


@router.post("/resend-verification")
async def resend_verification(
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    if user.email_verified:
        return {"success": True, "message": "Email already verified"}
    code = f"{secrets.randbelow(1000000):06d}"
    user.email_verify_code = hash_password(code)
    user.email_verify_expires = datetime.utcnow() + timedelta(hours=24)
    await db.commit()
    sent = send_verification_email(user.email, user.name, code)
    if not sent:
        log.info("Verification code for %s: %s (SMTP not configured)", user.email, code)
    return {"success": True, "message": "Verification code sent"}


# ═══════════════════════════════════════════════════════════════════════
# Two-Factor Authentication (TOTP)
# ═══════════════════════════════════════════════════════════════════════

@router.post("/2fa/setup")
async def two_factor_setup(
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    if not HAS_TOTP:
        raise HTTPException(status_code=501, detail="2FA not available (pyotp not installed)")
    if user.totp_enabled:
        raise HTTPException(status_code=400, detail="2FA is already enabled")

    secret = pyotp.random_base32()
    user.totp_secret = secret
    backup_codes = [secrets.token_hex(4) for _ in range(8)]
    user.backup_codes = [hash_password(c) for c in backup_codes]
    await db.commit()

    totp = pyotp.TOTP(secret)
    qr_url = totp.provisioning_uri(name=user.email, issuer_name="Satya Drishti")
    return {"secret": secret, "qr_url": qr_url, "backup_codes": backup_codes}


@router.post("/2fa/confirm")
async def two_factor_confirm(
    request: TwoFactorCodeRequest,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    if not HAS_TOTP:
        raise HTTPException(status_code=501, detail="2FA not available")
    if user.totp_enabled:
        raise HTTPException(status_code=400, detail="2FA is already enabled")
    if not user.totp_secret:
        raise HTTPException(status_code=400, detail="Call /2fa/setup first")

    totp = pyotp.TOTP(user.totp_secret)
    if not totp.verify(request.code, valid_window=1):
        raise HTTPException(status_code=400, detail="Invalid TOTP code")

    user.totp_enabled = True
    await db.commit()
    log.info("2FA enabled for %s", user.email)
    return {"success": True}


@router.post("/2fa/verify")
async def two_factor_verify(
    request: TwoFactorVerifyRequest,
    db: AsyncSession = Depends(get_db),
):
    """Verify TOTP code during login (using temp_token from login response)."""
    if not HAS_TOTP:
        raise HTTPException(status_code=501, detail="2FA not available")

    result = await db.execute(
        select(User).where(User.reset_token_expires > datetime.utcnow(), User.totp_enabled == True)
    )
    users = result.scalars().all()
    target_user = None
    for u in users:
        if u.reset_token_hash and verify_password(request.temp_token, u.reset_token_hash):
            target_user = u
            break

    if not target_user:
        raise HTTPException(status_code=401, detail="Invalid or expired temporary token")

    totp = pyotp.TOTP(target_user.totp_secret)
    code_valid = totp.verify(request.code, valid_window=1)

    if not code_valid and target_user.backup_codes:
        for i, hashed_backup in enumerate(target_user.backup_codes):
            if verify_password(request.code, hashed_backup):
                code_valid = True
                remaining = list(target_user.backup_codes)
                remaining.pop(i)
                target_user.backup_codes = remaining
                break

    if not code_valid:
        raise HTTPException(status_code=401, detail="Invalid 2FA code")

    target_user.reset_token_hash = None
    target_user.reset_token_expires = None
    await db.commit()

    token = create_access_token(target_user.id)
    return TokenResponse(token=token, user=_user_resp(target_user))


@router.post("/2fa/disable")
async def two_factor_disable(
    request: TwoFactorCodeRequest,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    if not HAS_TOTP:
        raise HTTPException(status_code=501, detail="2FA not available")
    if not user.totp_enabled:
        raise HTTPException(status_code=400, detail="2FA is not enabled")

    totp = pyotp.TOTP(user.totp_secret)
    if not totp.verify(request.code, valid_window=1):
        raise HTTPException(status_code=400, detail="Invalid TOTP code")

    user.totp_enabled = False
    user.totp_secret = None
    user.backup_codes = []
    await db.commit()
    log.info("2FA disabled for %s", user.email)
    return {"success": True}


# ═══════════════════════════════════════════════════════════════════════
# Right to Erasure (Account Deletion)
# ═══════════════════════════════════════════════════════════════════════

class DeleteAccountRequest(BaseModel):
    password: str = Field(..., min_length=1, max_length=128)
    confirm: str = Field(..., pattern=r"^DELETE$")


@router.delete("/account")
async def delete_account(
    request: DeleteAccountRequest,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """
    Right to erasure — permanently delete user account, all scans, cases,
    API keys, and archived media files. Irreversible.
    """
    # OAuth users don't have a real password — skip verification for them
    if not user.oauth_provider:
        if not verify_password(request.password, user.password_hash):
            raise HTTPException(status_code=403, detail="Incorrect password")

    user_email = user.email
    user_id = user.id

    # Delete all associated data
    from sqlalchemy import delete as sql_delete

    await db.execute(sql_delete(Scan).where(Scan.user_id == user_id))
    await db.execute(sql_delete(Case).where(Case.user_id == user_id))
    await db.execute(sql_delete(ApiKey).where(ApiKey.user_id == user_id))
    await db.delete(user)
    await db.commit()

    # Delete archived media files for this user
    from ..audit_archive import delete_user_archives
    deleted_files = delete_user_archives(user_id)

    log.info("Account deleted: %s (scans, cases, keys, %d archived files removed)", user_email, deleted_files)
    return {
        "success": True,
        "message": "Your account and all associated data have been permanently deleted.",
    }
