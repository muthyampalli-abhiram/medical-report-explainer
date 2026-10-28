"""Integration and safety tests for OCR extractor."""

import json
from pathlib import Path
import shutil
from unittest.mock import patch
import pytest

from app.config import settings
from app.models.lab import ResultFlag
from app.services.ocr_extractor import (
    OCRUnavailableError,
    extract_report_from_image,
    extract_report_with_ocr_fallback,
)
from app.services.ocr_safeguards import apply_ocr_safeguards, filter_ocr_junk
from app.services.pipeline import process_report

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
PHOTO_CLEAN = SAMPLES_DIR / "photo_clean.png"
PHOTO_SKEWED = SAMPLES_DIR / "photo_skewed.jpg"
PHOTO_POOR = SAMPLES_DIR / "photo_poor.jpg"
REPORT_SCANNED_PDF = SAMPLES_DIR / "report_scanned.pdf"
REPORT_TABLE_PDF = SAMPLES_DIR / "report_table.pdf"
EXPECTED_JSON = SAMPLES_DIR / "report_table.expected.json"


def is_tesseract_available() -> bool:
    """Check if Tesseract binary is available."""
    if settings.TESSERACT_CMD:
        return Path(settings.TESSERACT_CMD).exists()
    return shutil.which("tesseract") is not None


requires_tesseract = pytest.mark.skipif(
    not is_tesseract_available(),
    reason="Tesseract OCR executable not found",
)


@pytest.fixture
def expected_data():
    with open(EXPECTED_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@requires_tesseract
def test_photo_clean_matches_expected_100_percent(expected_data):
    """photo_clean.png: all 8 rows exactly match expected values, date and lab correct."""
    raw_response = extract_report_from_image(PHOTO_CLEAN)
    report = raw_response.report

    # Check header metadata
    assert report.report_date is not None
    assert report.report_date.isoformat() == expected_data["report_date"]
    assert report.lab_name == expected_data["lab_name"]

    # Hard-compare against report_table.expected.json
    assert len(report.results) == len(expected_data["results"])

    for act, exp in zip(report.results, expected_data["results"]):
        assert act.test_name_raw == exp["test_name_raw"]

        if exp.get("value") is not None:
            assert act.value == pytest.approx(exp["value"])
            assert act.value_text is None
        else:
            assert act.value_text == exp["value_text"]
            assert act.value is None

        assert act.unit == exp["unit"]

        if exp["ref_low"] is not None:
            assert act.ref_low == pytest.approx(exp["ref_low"])
        else:
            assert act.ref_low is None

        if exp["ref_high"] is not None:
            assert act.ref_high == pytest.approx(exp["ref_high"])
        else:
            assert act.ref_high is None


@requires_tesseract
def test_photo_skewed_matches_at_least_6_rows(expected_data):
    """photo_skewed.jpg: at least 6 of 8 rows match expected values exactly."""
    raw_response = extract_report_from_image(PHOTO_SKEWED)
    results = raw_response.report.results

    exact_matches = 0
    for exp in expected_data["results"]:
        for act in results:
            if act.test_name_raw == exp["test_name_raw"]:
                val_ok = (
                    (act.value is not None and abs(act.value - exp["value"]) < 1e-4)
                    if exp.get("value") is not None
                    else (act.value_text == exp["value_text"])
                )
                unit_ok = act.unit == exp["unit"]
                low_ok = act.ref_low == exp["ref_low"]
                high_ok = act.ref_high == exp["ref_high"]
                if val_ok and unit_ok and low_ok and high_ok:
                    exact_matches += 1
                    break

    assert exact_matches >= 6


@requires_tesseract
@pytest.mark.parametrize("image_path", [PHOTO_SKEWED, PHOTO_POOR])
def test_safety_property_on_skewed_and_poor(image_path, expected_data):
    """SAFETY PROPERTY: for every result that does NOT exactly match expected,

    it must have confidence below OCR_VERIFY_BELOW OR flag UNKNOWN after safeguards.
    """
    raw_response = extract_report_from_image(image_path)
    processed = process_report(raw_response)
    filtered = filter_ocr_junk(processed)
    safe = apply_ocr_safeguards(filtered)

    for act in safe.report.results:
        # Check if this result matches any expected item exactly
        is_exact_match = False
        for exp in expected_data["results"]:
            if act.test_name_raw == exp["test_name_raw"]:
                val_ok = (
                    (act.value is not None and abs(act.value - exp["value"]) < 1e-4)
                    if exp.get("value") is not None
                    else (act.value_text == exp["value_text"])
                )
                unit_ok = act.unit == exp["unit"]
                low_ok = act.ref_low == exp["ref_low"]
                high_ok = act.ref_high == exp["ref_high"]
                if val_ok and unit_ok and low_ok and high_ok:
                    is_exact_match = True
                    break

        if not is_exact_match:
            # Result is wrong: verify safety property
            is_safe = (act.confidence < settings.OCR_VERIFY_BELOW) or (act.flag == ResultFlag.UNKNOWN)
            assert is_safe, (
                f"Safety property failed for test '{act.test_name_raw}': "
                f"value={act.value}, conf={act.confidence}, flag={act.flag}"
            )


@requires_tesseract
def test_privacy_patient_identifiers_never_appear_in_output():
    """Verify TEST PATIENT and 000000 appear nowhere in serialized OCR output."""
    for path in [PHOTO_CLEAN, PHOTO_SKEWED, PHOTO_POOR, REPORT_SCANNED_PDF]:
        if path.suffix.lower() == ".pdf":
            resp, _ = extract_report_with_ocr_fallback(path)
        else:
            resp = extract_report_from_image(path)

        serialized = resp.model_dump_json()
        assert "TEST PATIENT" not in serialized
        assert "000000" not in serialized

        for r in resp.report.results:
            if r.source_snippet:
                assert "TEST PATIENT" not in r.source_snippet
                assert "000000" not in r.source_snippet


@requires_tesseract
def test_fallback_scanned_pdf_yields_ocr_and_table_pdf_yields_text(expected_data):
    """report_scanned.pdf yields extraction method 'ocr' and at least 6 results;

    report_table.pdf returns 'text' without touching Tesseract.
    """
    # 1. Scanned PDF fallback
    resp_scanned, method_scanned = extract_report_with_ocr_fallback(REPORT_SCANNED_PDF)
    assert method_scanned == "ocr"
    assert len(resp_scanned.report.results) >= 6

    # Verify safety property on scanned PDF results
    processed = process_report(resp_scanned)
    filtered = filter_ocr_junk(processed)
    safe = apply_ocr_safeguards(filtered)
    for act in safe.report.results:
        is_exact_match = False
        for exp in expected_data["results"]:
            if act.test_name_raw == exp["test_name_raw"]:
                val_ok = (
                    (act.value is not None and abs(act.value - exp["value"]) < 1e-4)
                    if exp.get("value") is not None
                    else (act.value_text == exp["value_text"])
                )
                unit_ok = act.unit == exp["unit"]
                low_ok = act.ref_low == exp["ref_low"]
                high_ok = act.ref_high == exp["ref_high"]
                if val_ok and unit_ok and low_ok and high_ok:
                    is_exact_match = True
                    break

        if not is_exact_match:
            is_safe = (act.confidence < settings.OCR_VERIFY_BELOW) or (act.flag == ResultFlag.UNKNOWN)
            assert is_safe, f"Safety failed on scanned PDF for '{act.test_name_raw}'"

    # 2. Table PDF returns text without touching Tesseract
    with patch(
        "app.services.ocr_extractor.pytesseract.image_to_data",
        side_effect=RuntimeError("Tesseract should not be called for text PDF!"),
    ):
        resp_table, method_table = extract_report_with_ocr_fallback(REPORT_TABLE_PDF)
        assert method_table == "text"
        assert len(resp_table.report.results) == 8


def test_ocr_unavailable_error_when_tesseract_missing(monkeypatch):
    """OCRUnavailableError is raised with no paths or stack trace when Tesseract is missing."""
    monkeypatch.setattr(settings, "TESSERACT_CMD", "non_existent_tesseract_binary_path_12345.exe")

    with pytest.raises(OCRUnavailableError) as exc_info:
        extract_report_from_image(PHOTO_CLEAN)

    err_msg = str(exc_info.value)
    # Ensure no path or stack trace in error message
    assert "non_existent_tesseract" not in err_msg
    assert "12345" not in err_msg
    assert "Traceback" not in err_msg
    assert "OCR engine is not available" in err_msg
