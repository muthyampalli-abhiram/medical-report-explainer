"""Pydantic schemas for plain-language medical report explanations."""

from pydantic import BaseModel, Field

from app.models.lab import ResultFlag


class TestExplanation(BaseModel):
    """Plain-language explanation for an individual laboratory test result."""

    test_name: str
    flag: ResultFlag
    result_sentence: str
    what_it_measures: str
    used_fallback: bool
    escalation: str | None = None


class ReportExplanation(BaseModel):
    """Overall plain-language explanation for an entire laboratory report."""

    report_id: str
    explanations: list[TestExplanation] = Field(default_factory=list)
    unexplained_tests: list[str] = Field(default_factory=list)
    overall_notice: str
    disclaimer: str
    language: str = "en"
    translation_reviewed: bool = True
    draft_notice: str | None = None
