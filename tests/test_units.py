"""Tests for unit conversion service."""

import pytest

from app.services.units import convert_to_canonical


def test_glucose_mmol_to_mgdl():
    """Verify 7.0 mmol/L glucose converts to approx 126.11 mg/dL."""
    res = convert_to_canonical("Fasting Glucose", 7.0, "mmol/L")
    assert res is not None
    val, unit = res
    assert val == pytest.approx(126.11, abs=0.01)
    assert unit == "mg/dL"


def test_total_cholesterol_mmol_to_mgdl():
    """Verify 5.2 mmol/L total cholesterol converts to approx 201.08 mg/dL."""
    res = convert_to_canonical("Total Cholesterol", 5.2, "mmol/L")
    assert res is not None
    val, unit = res
    assert val == pytest.approx(201.08, abs=0.01)
    assert unit == "mg/dL"


def test_triglycerides_mmol_to_mgdl():
    """Verify 1.7 mmol/L triglycerides converts to approx 150.57 mg/dL."""
    res = convert_to_canonical("Triglycerides", 1.7, "mmol/L")
    assert res is not None
    val, unit = res
    assert val == pytest.approx(150.57, abs=0.01)
    assert unit == "mg/dL"


def test_hemoglobin_g_per_l_to_g_per_dl():
    """Verify 130 g/L hemoglobin converts to 13.0 g/dL."""
    res = convert_to_canonical("Hemoglobin", 130.0, "g/L")
    assert res is not None
    val, unit = res
    assert val == pytest.approx(13.0, abs=0.01)
    assert unit == "g/dL"


def test_hba1c_mmol_per_mol_to_percent():
    """Verify HbA1c 53 mmol/mol converts to approx 7.0 %."""
    res = convert_to_canonical("HbA1c", 53.0, "mmol/mol")
    assert res is not None
    val, unit = res
    assert val == pytest.approx(7.0, abs=0.01)
    assert unit == "%"


def test_unrecognized_unit_returns_none():
    """Verify that an unrecognized or incompatible unit returns None."""
    assert convert_to_canonical("Fasting Glucose", 100.0, "liters") is None
    assert convert_to_canonical("Hemoglobin", 14.0, "invalid_unit") is None
    assert convert_to_canonical("Urine Protein", 10.0, "mg/dL") is None
    assert convert_to_canonical("Unknown Test", 50.0, "mg/dL") is None
