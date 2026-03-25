"""
Satya Drishti — Contact Routes
===============================
POST /api/contact — Submit a contact form (no auth required)
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import ContactSubmission

router = APIRouter(prefix="/api/contact", tags=["contact"])


class ContactRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    email: EmailStr
    subject: str = Field(..., min_length=1, max_length=500)
    message: str = Field(..., min_length=1, max_length=5000)


@router.post("/")
async def submit_contact(request: ContactRequest, db: AsyncSession = Depends(get_db)):
    submission = ContactSubmission(
        name=request.name,
        email=request.email,
        subject=request.subject,
        message=request.message,
    )
    db.add(submission)
    await db.commit()

    return {"success": True, "message": "Your message has been received. We will respond within 24 hours."}
