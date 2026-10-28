"""Unit tests for Pydantic v2 schemas in app.models.lab."""

from datetime import date
import pytest
from pydantic import ValidationError

from app.models.lab import ExtractionResponse, LabReport, LabResult, ResultFlag


def test_valid_numeric_result_accepted():
    """Verify that a valid numeric lab result is accepted."""
    result = LabResult(
        test_name_raw="Glycosylated Hb",
        test_name_canonical="HbA1c",
        value=5.6,
        unit="%",
        ref_low=4.0,
        ref_high=5.6,
        ref_text="4.0 - 5.6",
        flag=ResultFlag.NORMAL,
        source_page=1,
        source_snippet="Glycosylated Hb: 5.6 % (4.0 - 5.6)",
        confidence=1.0,
    )

    assert result.test_name_raw == "Glycosylated Hb"
    assert result.test_name_canonical == "HbA1c"
    assert result.value == 5.6
    assert result.value_text is None
    assert result.unit == "%"
    assert result.ref_low == 4.0
    assert result.ref_high == 5.6
    assert result.flag == ResultFlag.NORMAL
    assert result.confidence == 1.0


def test_valid_text_only_result_accepted():
    """Verify that a valid text-only result (e.g. 'Positive') is accepted."""
    result = LabResult(
        test_name_raw="Urine Ketones",
        value_text="Positive",
    )

    assert result.test_name_raw == "Urine Ketones"
    assert result.value is None
    assert result.value_text == "Positive"
    assert result.flag == ResultFlag.UNKNOWN
    assert result.confidence == 1.0


def test_result_with_neither_value_nor_value_text_rejected():
    """Verify that a result with neither value nor value_text is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        LabResult(test_name_raw="Fasting Blood Glucose")
    assert "At least one of 'value' or 'value_text' must be present" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        LabResult(test_name_raw="Fasting Blood Glucose", value=None, value_text=None)
    assert "At least one of 'value' or 'value_text' must be present" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        LabResult(test_name_raw="Fasting Blood Glucose", value=None, value_text="   ")
    assert "At least one of 'value' or 'value_text' must be present" in str(exc_info.value)


def test_ref_low_greater_than_ref_high_rejected():
    """Verify that ref_low greater than ref_high is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        LabResult(
            test_name_raw="Serum Calcium",
            value=9.5,
            ref_low=12.0,
            ref_high=10.5,
        )
    assert "ref_low" in str(exc_info.value)
    assert "ref_high" in str(exc_info.value)

    # Valid boundary condition: ref_low equal to ref_high
    valid_boundary = LabResult(
        test_name_raw="Threshold Test",
        value=5.0,
        ref_low=5.0,
        ref_high=5.0,
    )
    assert valid_boundary.ref_low == valid_boundary.ref_high


def test_confidence_above_one_rejected():
    """Verify that confidence above 1.0 is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        LabResult(
            test_name_raw="Serum Iron",
            value=80.0,
            confidence=1.05,
        )
    assert "confidence" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        LabResult(
            test_name_raw="Serum Iron",
            value=80.0,
            confidence=-0.1,
        )
    assert "confidence" in str(exc_info.value)

    # Valid boundary condition: confidence between 0.0 and 1.0
    valid_res = LabResult(
        test_name_raw="Serum Iron",
        value=80.0,
        confidence=0.85,
    )
    assert valid_res.confidence == 0.85


def test_two_lab_reports_get_different_report_ids():
    """Verify that two LabReport objects get different report_ids."""
    report1 = LabReport()
    report2 = LabReport()

    assert report1.report_id != report2.report_id
    assert isinstance(report1.report_id, str)
    assert isinstance(report2.report_id, str)
    assert len(report1.report_id) == 36  # Standard UUID format length


def test_lab_report_serializes_to_json_and_parses_back():
    """Verify that LabReport serializes to JSON and parses back to an equal object."""
    report = LabReport(
        report_date=date(2025, 3, 10),
        lab_name="Metro Diagnostics Center",
        source_filename="cbc_report_2025.pdf",
        results=[
            LabResult(
                test_name_raw="Glycosylated Hb",
                test_name_canonical="HbA1c",
                value=5.4,
                unit="%",
                ref_low=4.0,
                ref_high=5.6,
                ref_text="4.0 - 5.6",
                flag=ResultFlag.NORMAL,
                source_page=1,
                source_snippet="Glycosylated Hb: 5.4 % (4.0 - 5.6)",
                confidence=0.98,
            ),
            LabResult(
                test_name_raw="Urine Ketones",
                value_text="Positive",
                flag=ResultFlag.UNKNOWN,
                source_page=2,
                source_snippet="Ketones: Positive",
                confidence=0.95,
            ),
        ],
    )

    json_payload = report.model_dump_json()
    parsed_report = LabReport.model_validate_json(json_payload)

    assert report == parsed_report
    assert parsed_report.report_id == report.report_id
    assert parsed_report.report_date == date(2025, 3, 10)
    assert len(parsed_report.results) == 2
    assert parsed_report.results[0].test_name_canonical == "HbA1c"
    assert parsed_report.results[1].value_text == "Positive"


def test_extraction_response_and_privacy_guarantee():
    """Verify ExtractionResponse defaults and privacy exclusion on LabReport."""
    report = LabReport(
        results=[
            LabResult(test_name_raw="Glucose", value=100.0),
        ]
    )
    extraction = ExtractionResponse(report=report)
    assert extraction.report == report
    assert extraction.warnings == []

    # Verify no personal identifiers on LabReport
    report_fields = LabReport.model_fields.keys()
    assert "patient_name" not in report_fields
    assert "patient_id" not in report_fields
    assert "phone" not in report_fields
    assert "address" not in report_fields
