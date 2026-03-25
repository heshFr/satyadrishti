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
    db_label = DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL.split("///")[-1]
    log.info("Database initialized at %s", db_label)
