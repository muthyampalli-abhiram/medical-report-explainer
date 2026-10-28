"""Pydantic v2 schemas for medical lab reports and results."""

from datetime import date
from enum import Enum
from typing import Self
import uuid

from pydantic import BaseModel, Field, model_validator


class ResultFlag(str, Enum):
    """Flag classification for a lab result.

    Flags are computed downstream by deterministic code/rules, never by LLMs.
    Defaults to UNKNOWN for all newly extracted results.
    """


    

    NORMAL = "NORMAL"
    LOW = "LOW"
    HIGH = "HIGH"
    CRITICAL_LOW = "CRITICAL_LOW"
    CRITICAL_HIGH = "CRITICAL_HIGH"
    UNKNOWN = "UNKNOWN"


class LabResult(BaseModel):
    """Schema representing an individual laboratory test result."""

    test_name_raw: str
    test_name_canonical: str | None = None
    value: float | None = None
    value_text: str | None = None
    unit: str | None = None
    ref_low: float | None = None
    ref_high: float | None = None
    ref_text: str | None = None
    flag: ResultFlag = ResultFlag.UNKNOWN
    source_page: int | None = None
    source_snippet: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    value_original: float | None = None
    unit_original: str | None = None

    @model_validator(mode="after")
    def validate_result_and_reference_range(self) -> Self:
        """Validate result presence and reference range sanity."""
        if self.value is None and (self.value_text is None or not self.value_text.strip()):
            raise ValueError("At least one of 'value' or 'value_text' must be present.")

        if self.ref_low is not None and self.ref_high is not None and self.ref_low > self.ref_high:
            raise ValueError(
                f"ref_low ({self.ref_low}) must be less than or equal to ref_high ({self.ref_high})."
            )

        return self


class LabReport(BaseModel):
    """Structured representation of an extracted medical lab report.

    Note on Privacy:
    Personal identifiers (such as patient name, phone number, address, or patient ID)
    are intentionally excluded from this model to ensure patient confidentiality
    and adhere to privacy-by-design principles.
    """

    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    report_date: date | None = None
    lab_name: str | None = None
    source_filename: str | None = None
    results: list[LabResult] = Field(default_factory=list)


class ExtractionResponse(BaseModel):
    """Container for the extraction output along with any processing warnings."""

    report: LabReport
    warnings: list[str] = Field(default_factory=list)
