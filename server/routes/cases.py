"""
Satya Drishti — Case Management Routes
========================================
CRUD operations for investigation cases.

POST   /api/cases              — Create a new case
GET    /api/cases              — List user's cases (paginated)
GET    /api/cases/{id}         — Get single case detail
PUT    /api/cases/{id}         — Update case fields
DELETE /api/cases/{id}         — Delete a case
POST   /api/cases/{id}/scans   — Add a scan to a case
DELETE /api/cases/{id}/scans/{scan_id} — Remove a scan from a case
"""

from datetime import datetime

from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import Case, Scan, User
from ..auth import require_auth

router = APIRouter(prefix="/api/cases", tags=["cases"])


# ─── Pydantic Schemas ───


class CaseStatus(str, Enum):
    open = "open"
    investigating = "investigating"
    resolved = "resolved"


class CaseCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field("", max_length=5000)
    scan_ids: list[str] = Field(default_factory=list, max_length=100)


class CaseUpdateRequest(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500)
    description: str | None = Field(None, max_length=5000)
    status: CaseStatus | None = None


class CaseResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    title: str
    description: str
    scan_ids: list[str]
    status: str
    created_at: str
    updated_at: str


class PaginatedCasesResponse(BaseModel):
    items: list[CaseResponse]
    total: int
    page: int
    per_page: int


class AddScanRequest(BaseModel):
    scan_id: str


# ─── Endpoints ───


@router.post("/", response_model=CaseResponse, status_code=201)
async def create_case(
    request: CaseCreateRequest,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Create a new investigation case."""
    # Validate that all referenced scan_ids belong to this user
    if request.scan_ids:
        result = await db.execute(
            select(func.count(Scan.id)).where(
                Scan.id.in_(request.scan_ids),
                Scan.user_id == user.id,
            )
        )
        valid_count = result.scalar()
        if valid_count != len(request.scan_ids):
            raise HTTPException(status_code=400, detail="One or more scan IDs are invalid or do not belong to you")

    case = Case(
        user_id=user.id,
        title=request.title,
        description=request.description,
        scan_ids=request.scan_ids,
    )
    db.add(case)
    await db.commit()
    await db.refresh(case)

    return _case_to_response(case)


@router.get("/", response_model=PaginatedCasesResponse)
async def list_cases(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    status: CaseStatus | None = Query(None),
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """List the current user's cases with optional status filter."""
    query = select(Case).where(Case.user_id == user.id)
    count_query = select(func.count(Case.id)).where(Case.user_id == user.id)

    if status:
        query = query.where(Case.status == status.value)
        count_query = count_query.where(Case.status == status.value)

    count_result = await db.execute(count_query)
    total = count_result.scalar()

    offset = (page - 1) * per_page
    result = await db.execute(
        query.order_by(Case.updated_at.desc()).offset(offset).limit(per_page)
    )
    cases = result.scalars().all()

    return PaginatedCasesResponse(
        items=[_case_to_response(c) for c in cases],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(
    case_id: str,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Get a single case by ID."""
    case = await _get_user_case(case_id, user.id, db)
    return _case_to_response(case)


@router.put("/{case_id}", response_model=CaseResponse)
async def update_case(
    case_id: str,
    request: CaseUpdateRequest,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Update case fields (title, description, status)."""
    case = await _get_user_case(case_id, user.id, db)

    if request.title is not None:
        case.title = request.title
    if request.description is not None:
        case.description = request.description
    if request.status is not None:
        case.status = request.status.value

    case.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(case)

    return _case_to_response(case)


@router.delete("/{case_id}")
async def delete_case(
    case_id: str,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Delete a case."""
    case = await _get_user_case(case_id, user.id, db)
    await db.delete(case)
    await db.commit()
    return {"status": "deleted"}


@router.post("/{case_id}/scans", response_model=CaseResponse)
async def add_scan_to_case(
    case_id: str,
    request: AddScanRequest,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Add a scan to an existing case."""
    case = await _get_user_case(case_id, user.id, db)

    # Verify scan exists and belongs to user
    result = await db.execute(
        select(Scan).where(Scan.id == request.scan_id, Scan.user_id == user.id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Scan not found")

    scan_ids = list(case.scan_ids or [])
    if request.scan_id in scan_ids:
        raise HTTPException(status_code=409, detail="Scan already in this case")

    scan_ids.append(request.scan_id)
    case.scan_ids = scan_ids
    case.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(case)

    return _case_to_response(case)


@router.delete("/{case_id}/scans/{scan_id}", response_model=CaseResponse)
async def remove_scan_from_case(
    case_id: str,
    scan_id: str,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Remove a scan from a case."""
    case = await _get_user_case(case_id, user.id, db)

    scan_ids = list(case.scan_ids or [])
    if scan_id not in scan_ids:
        raise HTTPException(status_code=404, detail="Scan not in this case")

    scan_ids.remove(scan_id)
    case.scan_ids = scan_ids
    case.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(case)

    return _case_to_response(case)


# ─── Helpers ───


async def _get_user_case(case_id: str, user_id: str, db: AsyncSession) -> Case:
    """Fetch a case belonging to the user or raise 404."""
    result = await db.execute(
        select(Case).where(Case.id == case_id, Case.user_id == user_id)
    )
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


def _case_to_response(case: Case) -> CaseResponse:
    return CaseResponse(
        id=case.id,
        title=case.title,
        description=case.description or "",
        scan_ids=case.scan_ids or [],
        status=case.status,
        created_at=case.created_at.isoformat(),
        updated_at=case.updated_at.isoformat(),
    )
