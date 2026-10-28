"""Tests for rule-based result flagging and critical overrides."""

from app.models.lab import LabResult, ResultFlag
from app.services.flagger import compute_flag


def test_flagger_normal_range():
    """Verify normal result within bounds."""
    result = LabResult(
        test_name_raw="Fasting Glucose",
        test_name_canonical="Fasting Glucose",
        value=85.0,
        unit="mg/dL",
        ref_low=70.0,
        ref_high=99.0,
        ref_text="70 - 99",
    )
    flag, warning = compute_flag(result)
    assert flag == ResultFlag.NORMAL
    assert warning is None


def test_flagger_low_range():
    """Verify low result below ref_low."""
    result = LabResult(
        test_name_raw="Fasting Glucose",
        test_name_canonical="Fasting Glucose",
        value=65.0,
        unit="mg/dL",
        ref_low=70.0,
        ref_high=99.0,
        ref_text="70 - 99",
    )
    flag, warning = compute_flag(result)
    assert flag == ResultFlag.LOW
    assert warning is None


def test_flagger_high_range():
    """Verify high result above ref_high."""
    result = LabResult(
        test_name_raw="Fasting Glucose",
        test_name_canonical="Fasting Glucose",
        value=120.0,
        unit="mg/dL",
        ref_low=70.0,
        ref_high=99.0,
        ref_text="70 - 99",
    )
    flag, warning = compute_flag(result)
    assert flag == ResultFlag.HIGH
    assert warning is None


def test_flagger_strict_less_than_boundary():
    """Verify strict '< 200' with value 200 is flagged HIGH."""
    result = LabResult(
        test_name_raw="Total Cholesterol",
        test_name_canonical="Total Cholesterol",
        value=200.0,
        unit="mg/dL",
        ref_high=200.0,
        ref_text="< 200",
    )
    flag, warning = compute_flag(result)
    assert flag == ResultFlag.HIGH
    assert warning is None


def test_flagger_inclusive_less_than_or_equal_boundary():
    """Verify inclusive '<= 100' with value 100 is flagged NORMAL."""
    result = LabResult(
        test_name_raw="LDL Cholesterol",
        test_name_canonical="LDL Cholesterol",
        value=100.0,
        unit="mg/dL",
        ref_high=100.0,
        ref_text="<= 100",
    )
    flag, warning = compute_flag(result)
    assert flag == ResultFlag.NORMAL
    assert warning is None


def test_flagger_strict_greater_than_boundary():
    """Verify strict '> 40' with value 40 is flagged LOW."""
    result = LabResult(
        test_name_raw="HDL Cholesterol",
        test_name_canonical="HDL Cholesterol",
        value=40.0,
        unit="mg/dL",
        ref_low=40.0,
        ref_text="> 40",
    )
    flag, warning = compute_flag(result)
    assert flag == ResultFlag.LOW
    assert warning is None


def test_flagger_critical_glucose_low():
    """Verify critical glucose 35 is flagged CRITICAL_LOW."""
    result = LabResult(
        test_name_raw="Fasting Glucose",
        test_name_canonical="Fasting Glucose",
        value=35.0,
        unit="mg/dL",
        ref_low=70.0,
        ref_high=99.0,
        ref_text="70 - 99",
    )
    flag, warning = compute_flag(result)
    assert flag == ResultFlag.CRITICAL_LOW
    assert warning is None


def test_flagger_critical_glucose_high():
    """Verify critical glucose 450 is flagged CRITICAL_HIGH."""
    result = LabResult(
        test_name_raw="Fasting Glucose",
        test_name_canonical="Fasting Glucose",
        value=450.0,
        unit="mg/dL",
        ref_low=70.0,
        ref_high=99.0,
        ref_text="70 - 99",
    )
    flag, warning = compute_flag(result)
    assert flag == ResultFlag.CRITICAL_HIGH
    assert warning is None


def test_flagger_critical_potassium_without_printed_range():
    """Verify critical potassium 7.0 with no printed range is flagged CRITICAL_HIGH."""
    result = LabResult(
        test_name_raw="Potassium",
        test_name_canonical="Potassium",
        value=7.0,
        unit="mmol/L",
        ref_low=None,
        ref_high=None,
        ref_text=None,
    )
    flag, warning = compute_flag(result)
    assert flag == ResultFlag.CRITICAL_HIGH
    assert warning is None


def test_flagger_text_result_matches_reference():
    """Verify text result 'Negative' vs 'Negative' is flagged NORMAL."""
    result = LabResult(
        test_name_raw="Urine Protein",
        test_name_canonical="Urine Protein",
        value_text="Negative",
        ref_text="Negative",
    )
    flag, warning = compute_flag(result)
    assert flag == ResultFlag.NORMAL
    assert warning is None


def test_flagger_text_result_differs_from_reference_gives_warning():
    """Verify 'Positive' vs 'Negative' is flagged UNKNOWN with doctor consultation warning."""
    result = LabResult(
        test_name_raw="Urine Protein",
        test_name_canonical="Urine Protein",
        value_text="Positive",
        ref_text="Negative",
    )
    flag, warning = compute_flag(result)
    assert flag == ResultFlag.UNKNOWN
    assert warning == "Non-numeric result differs from the reference. Please review with your doctor."


def test_flagger_missing_reference_range_gives_unknown():
    """Verify result with no usable range and no critical override is flagged UNKNOWN."""
    result = LabResult(
        test_name_raw="Uncatalogued Biomarker",
        test_name_canonical=None,
        value=15.0,
        ref_low=None,
        ref_high=None,
        ref_text=None,
    )
    flag, warning = compute_flag(result)
    assert flag == ResultFlag.UNKNOWN
    assert warning is None
