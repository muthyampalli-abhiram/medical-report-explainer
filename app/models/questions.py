"""Models for doctor question generation based on lab results and biomarker trends."""

from pydantic import BaseModel, Field


class DoctorQuestion(BaseModel):
    """A single deterministic question for the patient to ask their doctor."""

    text: str
    related_test: str | None = None
    reason: str


class DoctorQuestions(BaseModel):
    """Collection of doctor questions generated for a lab report."""

    report_id: str
    questions: list[DoctorQuestion] = Field(default_factory=list)
    general_questions: list[str] = Field(default_factory=list)
    trend_notes: list[str] = Field(default_factory=list)
