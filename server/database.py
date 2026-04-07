"""
Satya Drishti — Database Layer
==============================
SQLAlchemy async engine with SQLite for local development.
"""

import logging

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from .config import DATABASE_URL

log = logging.getLogger("satyadrishti.database")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    from . import models  # noqa: F401 — ensure models are registered

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Auto-migrate: add missing columns to existing tables (SQLite safe)
    await _auto_migrate()

    db_label = DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL.split("///")[-1]
    log.info("Database initialized at %s", db_label)


async def _auto_migrate():
    """Add missing columns to existing tables. Safe to run multiple times."""
    from sqlalchemy import text, inspect as sa_inspect

    new_user_columns = {
        "email_verified": "BOOLEAN DEFAULT 0",
        "email_verify_code": "VARCHAR",
        "email_verify_expires": "DATETIME",
        "totp_secret": "VARCHAR",
        "totp_enabled": "BOOLEAN DEFAULT 0",
        "backup_codes": "JSON DEFAULT '[]'",
        "oauth_provider": "VARCHAR(20)",
        "notify_email_threats": "BOOLEAN DEFAULT 1",
        "notify_email_reports": "BOOLEAN DEFAULT 0",
        "notify_push_enabled": "BOOLEAN DEFAULT 0",
        "notify_push_subscription": "JSON",
        "emergency_contact_name": "VARCHAR(200)",
        "emergency_contact_phone": "VARCHAR(20)",
    }

    async with engine.begin() as conn:
        def _migrate(connection):
            insp = sa_inspect(connection)
            # Migrate users table
            if insp.has_table("users"):
                existing = {c["name"] for c in insp.get_columns("users")}
                for col, col_type in new_user_columns.items():
                    if col not in existing:
                        try:
                            connection.execute(text(f"ALTER TABLE users ADD COLUMN {col} {col_type}"))
                            log.info("Added column users.%s", col)
                        except Exception as e:
                            log.debug("Column users.%s may already exist: %s", col, e)

        await conn.run_sync(_migrate)
