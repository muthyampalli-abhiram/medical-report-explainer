"""Tests for medical safety guardrail checks."""

import pytest

from app.services.safety import check_text, safe_or_fallback


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "Based on this result, you have diabetes.",
        "It looks like you are diabetic and need care.",
        "Your low blood counts show that you are anemic.",
        "This test provides a definitive diagnosis of heart failure.",
        "This indicates kidney failure in your sample.",
        "The elevated level confirms hepatic cirrhosis.",
        "You appear to be suffering from severe hypertension.",
        "Please stop taking your blood pressure medication.",
        "You should start taking insulin immediately.",
        "Make sure you increase your dose to 50 mg.",
        "You need to decrease your dose right away.",
        "Please take 500 mg every morning.",
        "You should take aspirin every evening.",
        "There is no need for medication anymore.",
        "Feel free to skip your medication today.",
        "There is nothing to worry about at all.",
        "Don't worry about this elevated level.",
        "There is no need to see a doctor for this.",
        "This is a life-threatening medical emergency.",
        "If left untreated, you will die soon.",
    ],
)
def test_unsafe_strings_flagged(unsafe_text):
    """Verify at least 8 unsafe strings across diagnosis, medication, and alarm/reassurance are flagged."""
    violations = check_text(unsafe_text)
    assert len(violations) > 0

    fallback_text, used_fallback = safe_or_fallback(unsafe_text, "Safe fallback text.")
    assert used_fallback is True
    assert fallback_text == "Safe fallback text."


@pytest.mark.parametrize(
    "safe_text",
    [
        "HbA1c shows your average blood sugar over the last three months.",
        "Fasting glucose measures the concentration of sugar in your blood after fasting.",
        "Total cholesterol evaluates the combined amount of cholesterol lipids in the blood.",
        "Hemoglobin is an oxygen-carrying protein found inside red blood cells.",
        "Potassium is an electrolyte that supports normal heart and muscle rhythm.",
        "Urine protein checks for proteins filtered by the kidneys into urine.",
        "Doctors order this test to evaluate general metabolic and lipid balance.",
    ],
)
def test_safe_strings_pass(safe_text):
    """Verify safe informational descriptions pass guardrails without violations."""
    violations = check_text(safe_text)
    assert violations == []

    text, used_fallback = safe_or_fallback(safe_text, "Fallback.")
    assert used_fallback is False
    assert text == safe_text
