"""Models for deterministic prescription extraction and summarization."""

from pydantic import BaseModel, Field


class Medication(BaseModel):
    """An individual medication item extracted from a prescription."""

    name_as_written: str
    form: str | None = None
    strength: str | None = None
    frequency_raw: str | None = None
    frequency_plain: str | None = None
    duration_raw: str | None = None
    instructions_raw: str | None = None
    instructions_plain: str | None = None
    source_snippet: str


class PrescriptionSummary(BaseModel):
    """Deterministic summary of a prescription."""

    medications: list[Medication] = Field(default_factory=list)
    other_lines: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str
