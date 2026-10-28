"""API response and request schemas."""

from datetime import date
from pydantic import BaseModel, Field

from app.models.lab import LabResult


class UploadResponse(BaseModel):
    """Response schema for uploaded and processed medical lab reports."""

    report_id: str
    report_date: date | None = None
    lab_name: str | None = None
    result_count: int
    abnormal_count: int
    critical_count: int
    results: list[LabResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str
    extraction_method: str = "text"
    low_confidence_count: int = 0


class ErrorResponse(BaseModel):
    """Standard error response model."""

    detail: str
