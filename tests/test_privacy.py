"""Tests for PII masking and privacy defense in depth."""

from app.services.privacy import mask_pii


def test_pii_masking():
    """Verify masking of emails, phone numbers (+91 and 10-digit), Aadhaar numbers, and patient labels."""
    raw = (
        "Contact patient at john.doe@example.com or support@healthlab.org. "
        "Call +91 9876543210 or 9876543210 or +91-98765-43210. "
        "Aadhaar card: 1234 5678 9012 or 123456789012. "
        "Patient: John Doe. Name: Jane Smith. ID: 12345. MRN: MRN9988."
    )

    masked = mask_pii(raw)

    assert "john.doe@example.com" not in masked
    assert "support@healthlab.org" not in masked
    assert "9876543210" not in masked
    assert "1234 5678 9012" not in masked
    assert "123456789012" not in masked
    assert "John Doe" not in masked
    assert "Jane Smith" not in masked
    assert "[REDACTED]" in masked


def test_ordinary_medical_text_is_unchanged():
    """Verify ordinary clinical descriptions and reference ranges are preserved without changes."""
    ordinary_text = (
        "HbA1c shows your average blood sugar level over roughly the last three months. "
        "Fasting glucose is 110 mg/dL with reference range 70 - 99 mg/dL. "
        "Potassium level was measured at 4.2 mmol/L."
    )

    assert mask_pii(ordinary_text) == ordinary_text
