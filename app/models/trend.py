"""Trend models for historical lab results."""

from datetime import date
from pydantic import BaseModel

from app.models.lab import ResultFlag


class TrendPoint(BaseModel):
    """A single historical biomarker measurement over time."""

    report_id: str
    report_date: date
    value: float
    unit: str | None = None
    flag: ResultFlag
    ref_low: float | None = None
    ref_high: float | None = None
