"""Unit tests for OCR safeguards and plausibility rules."""

import pytest

from app.models.lab import ExtractionResponse, LabReport, LabResult, ResultFlag
from app.services.ocr_safeguards import apply_ocr_safeguards, filter_ocr_junk


def test_filter_ocr_junk_rules():
    """Test filter_ocr_junk rules on hand-built objects (no Tesseract):
    - junk names 'im mg/dl', '214 mg/d', '13.1 e/a', and 'Negative' are removed
    - plausible unrecognized name 'Vitamin D' with confidence 0.95 is kept
    - the same name with confidence 0.5 is removed
    - recognized tests are never removed, even at confidence 0.3
    - the skipped-lines warning has the right count and contains none of the removed text
    - values of kept results are unchanged.
    """
    results = [
        # 1. Junk row 1
        LabResult(
            test_name_raw="im mg/dl",
            test_name_canonical=None,
            value=7.0,
            confidence=0.46,
        ),
        # 2. Junk row 2
        LabResult(
            test_name_raw="214 mg/d",
            test_name_canonical=None,
            value=200.0,
            confidence=0.56,
        ),
        # 3. Junk row 3
        LabResult(
            test_name_raw="13.1 e/a",
            test_name_canonical=None,
            value=1.0,
            confidence=0.31,
        ),
        # 4. Junk row 4 (forbidden exact word)
        LabResult(
            test_name_raw="Negative",
            test_name_canonical=None,
            value=None,
            value_text="Negative",
            confidence=0.95,
        ),
        # 5. Plausible unrecognized with high confidence (kept)
        LabResult(
            test_name_raw="Vitamin D",
            test_name_canonical=None,
            value=45.2,
            unit="ng/mL",
            confidence=0.95,
        ),
        # 6. Plausible unrecognized with low confidence 0.5 (removed)
        LabResult(
            test_name_raw="Vitamin D",
            test_name_canonical=None,
            value=28.0,
            unit="ng/mL",
            confidence=0.50,
        ),
        # 7. Recognized test at low confidence 0.3 (never removed)
        LabResult(
            test_name_raw="HbA1c",
            test_name_canonical="HbA1c",
            value=7.4,
            unit="%",
            confidence=0.30,
        ),
    ]

    initial_warnings = [
        "Unrecognized test name: im mg/dl",
        "Unrecognized test name: 214 mg/d",
        "Unrecognized test name: 13.1 e/a",
        "Unrecognized test name: Negative",
        "Unrecognized test name: Vitamin D",
    ]

    extraction = ExtractionResponse(
        report=LabReport(results=results),
        warnings=list(initial_warnings),
    )

    filtered = filter_ocr_junk(extraction)

    # 5 results should have been removed:
    # 4 junk names + 1 low-confidence Vitamin D
    assert len(filtered.report.results) == 2

    kept_tests = [r.test_name_raw for r in filtered.report.results]
    assert "Vitamin D" in kept_tests
    assert "HbA1c" in kept_tests

    # Values of kept results are unchanged
    vit_d = next(r for r in filtered.report.results if r.test_name_raw == "Vitamin D")
    assert vit_d.value == 45.2
    assert vit_d.confidence == 0.95

    hba1c = next(r for r in filtered.report.results if r.test_name_raw == "HbA1c")
    assert hba1c.value == 7.4
    assert hba1c.confidence == 0.30

    # Skipped-lines warning has the right count (5)
    assert any("5 unreadable line(s) were skipped." in w for w in filtered.warnings)

    # Contains none of the removed text in any warning
    for w in filtered.warnings:
        assert "im mg/dl" not in w
        assert "214 mg/d" not in w
        assert "13.1 e/a" not in w
        assert "Negative" not in w


def test_confidence_low_below_unknown_threshold():
    """Confidence 0.5 (< 0.60) -> flag UNKNOWN and low-confidence warning."""
    res = LabResult(
        test_name_raw="HbA1c",
        test_name_canonical="HbA1c",
        value=5.4,
        unit="%",
        ref_low=4.0,
        ref_high=5.6,
        flag=ResultFlag.NORMAL,
        confidence=0.5,
    )
    extraction = ExtractionResponse(
        report=LabReport(results=[res]),
        warnings=[],
    )

    processed = apply_ocr_safeguards(extraction)

    assert processed.report.results[0].flag == ResultFlag.UNKNOWN
    assert processed.report.results[0].value == 5.4
    assert len(processed.warnings) == 1
    assert (
        "Low OCR confidence for HbA1c. The value could not be read reliably, "
        "so it has not been compared with the reference range. Please check it against your original report."
        in processed.warnings[0]
    )


def test_confidence_moderate_verify_warning():
    """Confidence 0.7 (0.60 <= conf < 0.85) -> flag kept and verify warning added."""
    res = LabResult(
        test_name_raw="HbA1c",
        test_name_canonical="HbA1c",
        value=5.4,
        unit="%",
        ref_low=4.0,
        ref_high=5.6,
        flag=ResultFlag.NORMAL,
        confidence=0.7,
    )
    extraction = ExtractionResponse(
        report=LabReport(results=[res]),
        warnings=[],
    )

    processed = apply_ocr_safeguards(extraction)

    assert processed.report.results[0].flag == ResultFlag.NORMAL
    assert processed.report.results[0].value == 5.4
    assert len(processed.warnings) == 1
    assert (
        "Please verify the value of HbA1c against your original report. The text may have been misread."
        in processed.warnings[0]
    )


def test_confidence_high_untouched():
    """Confidence 0.95 (>= 0.85) -> untouched, no warnings."""
    res = LabResult(
        test_name_raw="HbA1c",
        test_name_canonical="HbA1c",
        value=5.4,
        unit="%",
        ref_low=4.0,
        ref_high=5.6,
        flag=ResultFlag.NORMAL,
        confidence=0.95,
    )
    extraction = ExtractionResponse(
        report=LabReport(results=[res]),
        warnings=[],
    )

    processed = apply_ocr_safeguards(extraction)

    assert processed.report.results[0].flag == ResultFlag.NORMAL
    assert processed.report.results[0].value == 5.4
    assert len(processed.warnings) == 0


def test_implausible_value_sets_unknown_flag_and_warning():
    """HbA1c value 74 (implausible > 20) with confidence 0.99 -> UNKNOWN with implausible warning."""
    res = LabResult(
        test_name_raw="HbA1c",
        test_name_canonical="HbA1c",
        value=74.0,
        unit="%",
        ref_low=4.0,
        ref_high=5.6,
        flag=ResultFlag.HIGH,
        confidence=0.99,
    )
    extraction = ExtractionResponse(
        report=LabReport(results=[res]),
        warnings=[],
    )

    processed = apply_ocr_safeguards(extraction)

    assert processed.report.results[0].flag == ResultFlag.UNKNOWN
    # Value must never be modified
    assert processed.report.results[0].value == 74.0
    assert len(processed.warnings) == 1
    assert (
        "The value of HbA1c looks unlikely and may have been misread. "
        "Please check it against your original report."
        in processed.warnings[0]
    )


def test_value_never_modified_under_any_condition():
    """Values are never modified, digits never auto-corrected."""
    res = LabResult(
        test_name_raw="Potassium",
        test_name_canonical="Potassium",
        value=15.0,  # Implausible potassium > 9.0
        unit="mmol/L",
        ref_low=3.5,
        ref_high=5.0,
        flag=ResultFlag.CRITICAL_HIGH,
        confidence=0.45,
    )
    extraction = ExtractionResponse(
        report=LabReport(results=[res]),
        warnings=[],
    )

    processed = apply_ocr_safeguards(extraction)

    assert processed.report.results[0].value == 15.0
    assert processed.report.results[0].flag == ResultFlag.UNKNOWN


def test_unrecognized_test_name_handled_without_crashing():
    """Unrecognized test (test_name_canonical is None) is handled cleanly."""
    res = LabResult(
        test_name_raw="Some Exotic Marker 99",
        test_name_canonical=None,
        value=42.0,
        unit="ng/mL",
        flag=ResultFlag.UNKNOWN,
        confidence=0.75,
    )
    extraction = ExtractionResponse(
        report=LabReport(results=[res]),
        warnings=[],
    )

    processed = apply_ocr_safeguards(extraction)

    assert processed.report.results[0].value == 42.0
    assert processed.report.results[0].flag == ResultFlag.UNKNOWN
    assert len(processed.warnings) == 1
    assert "was not recognized" in processed.warnings[0]
    assert "Some Exotic Marker 99" in processed.warnings[0]
    assert "Please check it against your original report" in processed.warnings[0]


def test_regression_ocr_safeguard_unrecognized_test_names_set_unknown():
    """Regression test: unrecognized test names get ResultFlag.UNKNOWN and the specific warning,
    recognized test keeps flag, value of every result is unchanged, and text-layer cases keep today's behaviour.
    """
    from app.services.pipeline import process_report

    # 1. "Hb Aic" with confidence 0.91, value 7.4, range 4.0 - 5.6 -> UNKNOWN plus warning
    res1 = LabResult(
        test_name_raw="Hb Aic",
        test_name_canonical=None,
        value=7.4,
        unit="%",
        ref_low=4.0,
        ref_high=5.6,
        flag=ResultFlag.HIGH,
        confidence=0.91,
    )
    # 2. "HbAic" at 0.87
    res2 = LabResult(
        test_name_raw="HbAic",
        test_name_canonical=None,
        value=7.4,
        unit="%",
        ref_low=4.0,
        ref_high=5.6,
        flag=ResultFlag.HIGH,
        confidence=0.87,
    )
    # 3. "Hb Atc" at 0.912
    res3 = LabResult(
        test_name_raw="Hb Atc",
        test_name_canonical=None,
        value=7.4,
        unit="%",
        ref_low=4.0,
        ref_high=5.6,
        flag=ResultFlag.HIGH,
        confidence=0.912,
    )
    # 4. Recognized HbA1c at confidence 0.91
    res4 = LabResult(
        test_name_raw="HbA1c",
        test_name_canonical="HbA1c",
        value=7.4,
        unit="%",
        ref_low=4.0,
        ref_high=5.6,
        flag=ResultFlag.HIGH,
        confidence=0.91,
    )

    ext = ExtractionResponse(
        report=LabReport(results=[res1, res2, res3, res4]),
        warnings=[],
    )

    processed = apply_ocr_safeguards(ext)

    # Values must remain completely unchanged
    assert processed.report.results[0].value == 7.4
    assert processed.report.results[1].value == 7.4
    assert processed.report.results[2].value == 7.4
    assert processed.report.results[3].value == 7.4

    # Unrecognized names get UNKNOWN
    assert processed.report.results[0].flag == ResultFlag.UNKNOWN
    assert processed.report.results[1].flag == ResultFlag.UNKNOWN
    assert processed.report.results[2].flag == ResultFlag.UNKNOWN

    # Recognized HbA1c keeps its flag
    assert processed.report.results[3].flag == ResultFlag.HIGH

    # Check warnings for each unrecognized result
    assert any('The test name "Hb Aic" was not recognized, so this value was not compared with the reference range. Please check it against your original report.' in w for w in processed.warnings)
    assert any('The test name "HbAic" was not recognized, so this value was not compared with the reference range. Please check it against your original report.' in w for w in processed.warnings)
    assert any('The test name "Hb Atc" was not recognized, so this value was not compared with the reference range. Please check it against your original report.' in w for w in processed.warnings)

    # 5. Text-layer case: does not go through OCR safeguards, unrecognized name keeps today's behaviour
    text_res = LabResult(
        test_name_raw="Some Unrecognized Lab Test",
        value=150.0,
        unit="mg/dL",
        ref_low=70.0,
        ref_high=100.0,
    )
    text_ext = ExtractionResponse(
        report=LabReport(results=[text_res]),
        warnings=[],
    )
    text_processed = process_report(text_ext)
    assert text_processed.report.results[0].test_name_canonical is None
    assert text_processed.report.results[0].flag == ResultFlag.HIGH
    assert text_processed.report.results[0].value == 150.0

