"""
Satya Drishti — ORM Models
==========================
Database models for Users, Scans, Cases, API Keys, Notification Preferences,
and ContactSubmissions.
"""

import secrets
import uuid
from datetime import datetime

from sqlalchemy import Column, String, Float, Text, DateTime, Boolean, Integer, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship

from .database import Base


def generate_uuid():
    return str(uuid.uuid4())


def generate_api_key():
    return f"sd_{secrets.token_urlsafe(32)}"


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    name = Column(String(200), nullable=False)
    language_pref = Column(String(10), default="en")
    email_verified = Column(Boolean, default=False)
    email_verify_code = Column(String, nullable=True)
    email_verify_expires = Column(DateTime, nullable=True)
    reset_token_hash = Column(String, nullable=True)
    reset_token_expires = Column(DateTime, nullable=True)
    # 2FA
    totp_secret = Column(String, nullable=True)
    totp_enabled = Column(Boolean, default=False)
    backup_codes = Column(JSON, default=lambda: [])
    # OAuth
    oauth_provider = Column(String(20), nullable=True)  # "google", "github", or None
    # Notification Preferences
    notify_email_threats = Column(Boolean, default=True)
    notify_email_reports = Column(Boolean, default=False)
    notify_push_enabled = Column(Boolean, default=False)
    notify_push_subscription = Column(JSON, nullable=True)
    emergency_contact_name = Column(String(200), nullable=True)
    emergency_contact_phone = Column(String(20), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    scans = relationship("Scan", back_populates="user")
    cases = relationship("Case", back_populates="user")
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")


class ApiKey(Base):
    __tablename__ = "api_keys"
    __table_args__ = (
        Index("ix_api_keys_user_id", "user_id"),
        Index("ix_api_keys_key_hash", "key_hash"),
    )

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    key_hash = Column(String, nullable=False)  # bcrypt hash of the actual key
    key_prefix = Column(String(12), nullable=False)  # "sd_xxxx..." for display
    is_active = Column(Boolean, default=True)
    last_used = Column(DateTime, nullable=True)
    request_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="api_keys")


class Scan(Base):
    __tablename__ = "scans"
    __table_args__ = (
        Index("ix_scans_user_id", "user_id"),
    )

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    file_name = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    verdict = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    forensic_data = Column(JSON, default=lambda: [])
    raw_scores = Column(JSON, default=lambda: {})
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="scans")


class Case(Base):
    __tablename__ = "cases"
    __table_args__ = (
        Index("ix_cases_user_id", "user_id"),
        Index("ix_cases_status", "status"),
    )

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text, default="")
    scan_ids = Column(JSON, default=lambda: [])
    status = Column(String(20), default="open")  # open, investigating, resolved
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="cases")


class ContactSubmission(Base):
    __tablename__ = "contact_submissions"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String(200), nullable=False)
    email = Column(String(320), nullable=False)
    subject = Column(String(500), nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
