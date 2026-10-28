"""Privacy protection service for masking personally identifiable information."""

import re

# Email addresses
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Phone numbers (Indian +91, 10-digit mobile starting with 6-9, standard formatted)
PHONE_RE = re.compile(
    r"(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}\b|(?:\+91[\s-]?)?[6-9]\d{9}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"
)

# 12-digit Aadhaar-like numbers with or without spaces
AADHAAR_RE = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b|\b\d{12}\b")

# Header labels followed by values (Patient:, Name:, ID:, MRN:)
HEADER_IDENTIFIER_RE = re.compile(
    r"\b(?i:(?:Patient|Name|ID|MRN))\s*:\s*([^,\n\r;]+)"
)


def mask_pii(text: str) -> str:
    """Mask personally identifiable information (PII) with [REDACTED].

    Replaces emails, phone numbers (+91, 10-digit), 12-digit Aadhaar numbers,
    and strings following labels like Patient:, Name:, ID:, MRN:.
    """
    if not text or not isinstance(text, str):
        return ""

    result = EMAIL_RE.sub("[REDACTED]", text)
    result = PHONE_RE.sub("[REDACTED]", result)
    result = AADHAAR_RE.sub("[REDACTED]", result)

    def _replace_header(match: re.Match) -> str:
        label = match.group(0).split(":", 1)[0]
        return f"{label}: [REDACTED]"

    result = HEADER_IDENTIFIER_RE.sub(_replace_header, result)
    return result
