"""
Satya Drishti — Scan History Routes
====================================
GET    /api/scans          — List user's scans (paginated)
GET    /api/scans/{id}     — Get single scan detail
DELETE /api/scans/{id}     — Delete a scan
GET    /api/scans/{id}/report — Download PDF evidence report
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import Scan, User
from ..auth import require_auth
from ..report_generator import generate_report

router = APIRouter(prefix="/api/scans", tags=["scans"])


class ScanResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    file_name: str
    file_type: str
    verdict: str
    confidence: float
    forensic_data: list
    raw_scores: dict
    created_at: str


class PaginatedScansResponse(BaseModel):
    items: list[ScanResponse]
    total: int
    page: int
    per_page: int


@router.get("/", response_model=PaginatedScansResponse)
async def list_scans(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * per_page

    # Get total count
    count_result = await db.execute(select(func.count(Scan.id)).where(Scan.user_id == user.id))
    total = count_result.scalar()

    # Get paginated results
    result = await db.execute(
        select(Scan)
        .where(Scan.user_id == user.id)
        .order_by(Scan.created_at.desc())
        .offset(offset)
        .limit(per_page)
    )
    scans = result.scalars().all()

    return PaginatedScansResponse(
        items=[
            ScanResponse(
                id=s.id,
                file_name=s.file_name,
                file_type=s.file_type,
                verdict=s.verdict,
                confidence=s.confidence,
                forensic_data=s.forensic_data or [],
                raw_scores=s.raw_scores or {},
                created_at=s.created_at.isoformat(),
            )
            for s in scans
        ],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/{scan_id}", response_model=ScanResponse)
async def get_scan(scan_id: str, user: User = Depends(require_auth), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Scan).where(Scan.id == scan_id, Scan.user_id == user.id))
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    return ScanResponse(
        id=scan.id,
        file_name=scan.file_name,
        file_type=scan.file_type,
        verdict=scan.verdict,
        confidence=scan.confidence,
        forensic_data=scan.forensic_data or [],
        raw_scores=scan.raw_scores or {},
        created_at=scan.created_at.isoformat(),
    )


@router.delete("/{scan_id}")
async def delete_scan(scan_id: str, user: User = Depends(require_auth), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Scan).where(Scan.id == scan_id, Scan.user_id == user.id))
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    await db.delete(scan)
    await db.commit()
    return {"status": "deleted"}


@router.get("/{scan_id}/report")
async def download_report(scan_id: str, user: User = Depends(require_auth), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Scan).where(Scan.id == scan_id, Scan.user_id == user.id))
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    # Generate PDF in thread to avoid blocking the event loop
    pdf_bytes = await asyncio.to_thread(
        generate_report,
        scan_id=scan.id,
        file_name=scan.file_name,
        verdict=scan.verdict,
        confidence=scan.confidence,
        forensic_data=scan.forensic_data or [],
        created_at=scan.created_at.isoformat(),
    )

    from io import BytesIO

    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="satyadrishti_report_{scan_id[:8]}.pdf"'},
    )
