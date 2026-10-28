"""Tests for plain-language explanation generation, privacy isolation, and guardrails."""

from pathlib import Path
import pytest

from app.models.lab import ExtractionResponse, LabReport, LabResult, ResultFlag
from app.services.explainer import (
    CRITICAL_ESCALATION,
    CRITICAL_OVERALL_NOTICE,
    FIXED_DISCLAIMER,
    explain_report,
)
from app.services.llm_client import FakeLLMClient
from app.services.pdf_extractor import extract_report_from_pdf
from app.services.pipeline import process_report

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def test_prompt_privacy_no_report_metadata_or_values_sent_to_llm():
    """Verify that the LLM prompt contains ONLY canonical names and descriptions, no report data."""
    raw_extraction = extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf"))
    processed = process_report(raw_extraction)

    fake_llm = FakeLLMClient(response_or_fn={"tests": []})
    explain_report(processed, fake_llm)

    assert len(fake_llm.calls) == 1
    call = fake_llm.calls[0]
    user_prompt = call["user_prompt"]

    # Canonical names and descriptions must be in prompt
    assert "HbA1c" in user_prompt
    assert "Fasting Glucose" in user_prompt
    assert "average blood sugar" in user_prompt

    # Strictly verify NO report values, dates, lab names, filenames, or snippets are sent
    forbidden_strings = [
        "7.4",
        "132",
        "214",
        "Sample Diagnostics Lab",
        "report_table",
        "2026-01-15",
        "HbA1c | 7.4 | %",
    ]
    for forbidden in forbidden_strings:
        assert forbidden not in user_prompt, f"Forbidden string '{forbidden}' leaked into user prompt!"


def test_flag_and_result_sentence_immutable_regardless_of_llm_claims():
    """Verify result_sentence and flags are computed by code only and never altered by the LLM."""
    raw_extraction = extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf"))
    processed = process_report(raw_extraction)

    # Deceptive LLM claiming false normality
    deceptive_llm = FakeLLMClient(
        response_or_fn={
            "tests": [
                {
                    "name": "HbA1c",
                    "what_it_measures": "Your blood sugar is completely normal and healthy.",
                }
            ]
        }
    )

    explanation = explain_report(processed, deceptive_llm)
    hba1c_exp = next(e for e in explanation.explanations if e.test_name == "HbA1c")

    # Flag and result sentence must remain HIGH and deterministic
    assert hba1c_exp.flag == ResultFlag.HIGH
    assert (
        hba1c_exp.result_sentence
        == "Your result is 7.4 %. The lab's reference range is 4.0 - 5.6 %. This is above the reference range."
    )


def test_unsafe_llm_output_triggers_catalog_fallback():
    """Verify an unsafe diagnosis string triggers used_fallback=True and uses the catalog description."""
    raw_extraction = extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf"))
    processed = process_report(raw_extraction)

    unsafe_llm = FakeLLMClient(
        response_or_fn={
            "tests": [
                {
                    "name": "HbA1c",
                    "what_it_measures": "You have diabetes and need immediate treatment.",
                }
            ]
        }
    )

    explanation = explain_report(processed, unsafe_llm)
    hba1c_exp = next(e for e in explanation.explanations if e.test_name == "HbA1c")

    assert hba1c_exp.used_fallback is True
    assert "average blood sugar" in hba1c_exp.what_it_measures
    assert "You have diabetes" not in hba1c_exp.what_it_measures


def test_llm_failures_fall_back_cleanly_without_raising():
    """Verify exceptions, invalid JSON, and missing tests all fall back cleanly."""
    raw_extraction = extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf"))
    processed = process_report(raw_extraction)

    # 1. LLM raising an exception
    def _raise_error(s, u):
        raise RuntimeError("API connection timeout")

    error_llm = FakeLLMClient(response_or_fn=_raise_error)
    exp1 = explain_report(processed, error_llm)
    assert all(e.used_fallback for e in exp1.explanations)

    # 2. LLM returning empty or invalid structure
    invalid_llm = FakeLLMClient(response_or_fn={"unexpected": "structure"})
    exp2 = explain_report(processed, invalid_llm)
    assert all(e.used_fallback for e in exp2.explanations)

    # 3. LLM returning JSON missing tests
    partial_llm = FakeLLMClient(
        response_or_fn={
            "tests": [
                {"name": "HbA1c", "what_it_measures": "HbA1c measures average blood sugar."}
            ]
        }
    )
    exp3 = explain_report(processed, partial_llm)
    hba1c = next(e for e in exp3.explanations if e.test_name == "HbA1c")
    glucose = next(e for e in exp3.explanations if e.test_name == "Fasting Glucose")
    assert hba1c.used_fallback is False
    assert glucose.used_fallback is True


def test_critical_flag_produces_fixed_escalation_and_overall_notice():
    """Verify critical results receive the fixed escalation text and critical overall notice."""
    report = LabReport(
        results=[
            LabResult(
                test_name_raw="Potassium",
                test_name_canonical="Potassium",
                value=7.2,
                unit="mmol/L",
                ref_low=3.5,
                ref_high=5.0,
                flag=ResultFlag.CRITICAL_HIGH,
            )
        ]
    )
    response = ExtractionResponse(report=report)
    fake_llm = FakeLLMClient()

    explanation = explain_report(response, fake_llm)
    k_exp = explanation.explanations[0]

    assert k_exp.escalation == CRITICAL_ESCALATION
    assert explanation.overall_notice == CRITICAL_OVERALL_NOTICE


def test_unrecognized_test_name_goes_to_unexplained_tests_without_llm_call():
    """Verify unrecognized test names bypass LLM calls and go into unexplained_tests."""
    report = LabReport(
        results=[
            LabResult(
                test_name_raw="Uncatalogued Enzyme X",
                test_name_canonical=None,
                value=50.0,
                unit="U/L",
                flag=ResultFlag.UNKNOWN,
            )
        ]
    )
    response = ExtractionResponse(report=report)
    fake_llm = FakeLLMClient()

    explanation = explain_report(response, fake_llm)

    assert "Uncatalogued Enzyme X" in explanation.unexplained_tests
    assert len(explanation.explanations) == 0
    assert len(fake_llm.calls) == 0  # No LLM call made


def test_disclaimer_present_in_every_output():
    """Verify the medical disclaimer is always attached to report explanations."""
    raw_extraction = extract_report_from_pdf(str(SAMPLES_DIR / "report_lines.pdf"))
    processed = process_report(raw_extraction)

    explanation = explain_report(processed, FakeLLMClient())
    assert explanation.disclaimer == FIXED_DISCLAIMER


def test_full_pipeline_on_report_table():
    """Verify full end-to-end extraction, pipeline, and explanation on report_table.pdf."""
    raw_extraction = extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf"))
    processed = process_report(raw_extraction)

    fake_llm = FakeLLMClient()
    explanation = explain_report(processed, fake_llm)

    assert len(explanation.explanations) == 8

    hba1c_exp = next(e for e in explanation.explanations if e.test_name == "HbA1c")
    expected_sentence = (
        "Your result is 7.4 %. The lab's reference range is 4.0 - 5.6 %. This is above the reference range."
    )
    assert hba1c_exp.result_sentence == expected_sentence
