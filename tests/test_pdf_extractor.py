"""Tests for rule-based PDF extraction and reference range parsing."""

import json
from pathlib import Path
import pytest

from app.models.lab import ResultFlag
from app.services.pdf_extractor import (
    extract_report_from_pdf,
    parse_reference_range,
)

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def test_extract_report_table_matches_expected():
    """Verify 100% field-level match against report_table.expected.json."""
    pdf_path = SAMPLES_DIR / "report_table.pdf"
    expected_path = SAMPLES_DIR / "report_table.expected.json"

    with open(expected_path, "r", encoding="utf-8") as f:
        expected = json.load(f)

    response = extract_report_from_pdf(str(pdf_path))
    report = response.report

    # Assert report-level metadata
    assert report.report_date.isoformat() == expected["report_date"]
    assert report.lab_name == expected["lab_name"]
    assert len(report.results) == len(expected["results"])

    # 100% field-level match on results
    for actual, exp in zip(report.results, expected["results"]):
        assert actual.test_name_raw == exp["test_name_raw"]

        if "value" in exp and exp["value"] is not None:
            assert actual.value == pytest.approx(exp["value"])
            assert actual.value_text is None
        elif "value_text" in exp and exp["value_text"] is not None:
            assert actual.value_text == exp["value_text"]
            assert actual.value is None

        assert actual.unit == exp["unit"]

        if exp["ref_low"] is not None:
            assert actual.ref_low == pytest.approx(exp["ref_low"])
        else:
            assert actual.ref_low is None

        if exp["ref_high"] is not None:
            assert actual.ref_high == pytest.approx(exp["ref_high"])
        else:
            assert actual.ref_high is None

        # Verify invariants required by schemas
        assert actual.flag == ResultFlag.UNKNOWN
        assert actual.test_name_canonical is None
        assert actual.source_page == 1
        assert actual.source_snippet is not None and len(actual.source_snippet) > 0


def test_extract_report_lines_matches_expected():
    """Verify 100% field-level match against report_lines.expected.json."""
    pdf_path = SAMPLES_DIR / "report_lines.pdf"
    expected_path = SAMPLES_DIR / "report_lines.expected.json"

    with open(expected_path, "r", encoding="utf-8") as f:
        expected = json.load(f)

    response = extract_report_from_pdf(str(pdf_path))
    report = response.report

    # Assert report-level metadata
    assert report.report_date.isoformat() == expected["report_date"]
    assert report.lab_name == expected["lab_name"]
    assert len(report.results) == len(expected["results"])

    # 100% field-level match on results
    for actual, exp in zip(report.results, expected["results"]):
        assert actual.test_name_raw == exp["test_name_raw"]

        if "value" in exp and exp["value"] is not None:
            assert actual.value == pytest.approx(exp["value"])
            assert actual.value_text is None
        elif "value_text" in exp and exp["value_text"] is not None:
            assert actual.value_text == exp["value_text"]
            assert actual.value is None

        assert actual.unit == exp["unit"]

        if exp["ref_low"] is not None:
            assert actual.ref_low == pytest.approx(exp["ref_low"])
        else:
            assert actual.ref_low is None

        if exp["ref_high"] is not None:
            assert actual.ref_high == pytest.approx(exp["ref_high"])
        else:
            assert actual.ref_high is None

        # Verify invariants required by schemas
        assert actual.flag == ResultFlag.UNKNOWN
        assert actual.test_name_canonical is None
        assert actual.source_page == 1
        assert actual.source_snippet is not None and len(actual.source_snippet) > 0


def test_report_scanned_returns_zero_results_and_ocr_warning():
    """Verify scanned/image PDF returns zero results and required OCR warning."""
    scanned_path = SAMPLES_DIR / "report_scanned.pdf"
    response = extract_report_from_pdf(str(scanned_path))

    assert len(response.report.results) == 0
    assert "No text layer found. OCR is required for scanned documents." in response.warnings


def test_privacy_patient_identifiers_never_appear_in_output():
    """Verify patient name and ID never appear anywhere in serialized output."""
    sample_files = ["report_table.pdf", "report_lines.pdf", "report_scanned.pdf"]

    for filename in sample_files:
        pdf_path = SAMPLES_DIR / filename
        response = extract_report_from_pdf(str(pdf_path))
        serialized = response.model_dump_json()

        assert "TEST PATIENT" not in serialized
        assert "000000" not in serialized

        for result in response.report.results:
            if result.source_snippet:
                assert "TEST PATIENT" not in result.source_snippet
                assert "000000" not in result.source_snippet

        for warning in response.warnings:
            assert "TEST PATIENT" not in warning
            assert "000000" not in warning


@pytest.mark.parametrize(
    "input_ref, expected_low, expected_high",
    [
        ("4.0 - 5.6", 4.0, 5.6),
        ("4.0-5.6", 4.0, 5.6),
        ("70 - 99", 70.0, 99.0),
        ("13.0 - 17.0", 13.0, 17.0),
        ("4.0 – 5.6", 4.0, 5.6),
        ("< 200", None, 200.0),
        ("<200", None, 200.0),
        ("< 100", None, 100.0),
        ("< 150", None, 150.0),
        ("<= 100", None, 100.0),
        ("<=100", None, 100.0),
        ("> 40", 40.0, None),
        (">40", 40.0, None),
        (">= 50", 50.0, None),
        (">=50", 50.0, None),
        (">= 40.5", 40.5, None),
        ("Negative", None, None),
        ("Positive", None, None),
        ("Normal", None, None),
        ("Trace", None, None),
        ("garbage input", None, None),
        ("???", None, None),
        ("", None, None),
        (None, None, None),
    ],
)
def test_parse_reference_range_all_formats_and_garbage(input_ref, expected_low, expected_high):
    """Verify reference-range parser across all supported formats and garbage inputs."""
    low, high = parse_reference_range(input_ref)
    assert low == expected_low
    assert high == expected_high
