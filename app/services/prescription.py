"""Deterministic prescription parsing and summarization service."""

import json
from pathlib import Path
import re
from typing import Any

from app.models.prescription import Medication, PrescriptionSummary
from app.services.privacy import EMAIL_RE, PHONE_RE, mask_pii

# Form recognition pattern (matches optional list number followed by form word)
FORM_PATTERN = re.compile(
    r"^(?:\d+[\.\)\-:]\s*|\d+\s+)?(Tab(?:let)?|Cap(?:sule)?|Syp(?:rup)?|Inj(?:ection)?|Drops?|Oint(?:ment)?|Cream)\b",
    re.IGNORECASE,
)

# Header identifiers to drop completely (word-boundary matching on non-medication lines)
DROPPED_HEADER_PATTERN = re.compile(
    r"^(?:dr\.?|patient|name|id|mrn|age|phone|ph)\b\s*[:.\-]?",
    re.IGNORECASE,
)

# Strength pattern: e.g. 500 mg, 10 mg, 650 mg, 5 ml, 0.5 mg, 0.5%
STRENGTH_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|l|iu|%))(?!\w)",
    re.IGNORECASE,
)

# Duration pattern: e.g. x 30 days, for 2 weeks, x 1 month, x 14 days
DURATION_PATTERN = re.compile(
    r"\b((?:x|for)\s+\d+(?:\.\d+)?\s*(?:days?|weeks?|months?|d|w|m))\b",
    re.IGNORECASE,
)

# Dash frequency pattern: e.g. 1-0-1, 0-0-1, 1-1-1, 1/2-0-1/2, 0.5-0-0.5
DASH_FREQ_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+|\/\d+)?-\d+(?:\.\d+|\/\d+)?-\d+(?:\.\d+|\/\d+)?)\b"
)

# Known frequency abbreviation pattern
KNOWN_FREQ_PATTERN = re.compile(
    r"\b(OD|BD|BID|TDS|TID|QID|HS|SOS|STAT|Q8H|Q12H)\b",
    re.IGNORECASE,
)

# Generic candidate frequency token (short alphanumeric word)
CANDIDATE_FREQ_TOKEN = re.compile(
    r"\b([A-Za-z0-9]{2,5})\b"
)


def load_prescription_terms() -> dict[str, Any]:
    """Load prescription abbreviation tables from package data directory."""
    terms_path = Path(__file__).resolve().parent.parent / "data" / "prescription_terms.json"
    with open(terms_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_frequency_plain(freq_raw: str | None, frequencies_table: dict[str, str]) -> str | None:
    """Convert raw frequency into plain language using fixed abbreviation and pattern rules."""
    if not freq_raw or not isinstance(freq_raw, str):
        return None

    cleaned = freq_raw.strip()
    upper = cleaned.upper()

    # Look up in abbreviation table
    if upper in frequencies_table:
        return frequencies_table[upper]

    # Pattern frequencies like "1-0-1" (morning - afternoon - night)
    dash_match = re.match(r"^(\S+)-(\S+)-(\S+)$", cleaned)
    if dash_match:
        m, a, n = dash_match.groups()
        parts: list[str] = []
        if m not in ("0", "0.0"):
            parts.append(f"{m} in the morning")
        if a not in ("0", "0.0"):
            parts.append(f"{a} in the afternoon")
        if n not in ("0", "0.0"):
            parts.append(f"{n} at night")

        if len(parts) == 3:
            return f"{parts[0]}, {parts[1]}, and {parts[2]}"
        elif len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        elif len(parts) == 1:
            return parts[0]

    return None


def _should_drop_line(line: str) -> bool:
    """Check if non-medication line contains PII headers, phone numbers, or emails that must be dropped completely."""
    if not line or not line.strip():
        return True

    # Drop lines matching patient/doctor/identification headers
    if DROPPED_HEADER_PATTERN.search(line):
        return True

    # Drop lines containing phone numbers or emails
    if EMAIL_RE.search(line) or PHONE_RE.search(line):
        return True

    # Standalone 10-digit number (common phone format)
    if re.search(r"\b\d{10}\b", line):
        return True

    return False


def _parse_medication_line(
    line: str,
    terms: dict[str, Any],
) -> tuple[Medication, str | None]:
    """Parse a single medication line into a Medication object and optional warning."""
    source_snippet = line.strip()
    form_match = FORM_PATTERN.search(source_snippet)
    if not form_match:
        # Fallback if called erroneously
        return (
            Medication(name_as_written=mask_pii(source_snippet), source_snippet=mask_pii(source_snippet)),
            f"Some details on this line could not be read: {mask_pii(source_snippet)}. Please check with your pharmacist.",
        )

    form_as_written = form_match.group(1)
    # Text after form word
    rest = source_snippet[form_match.end():].strip()

    # 1. Extract Duration
    duration_raw: str | None = None
    dur_match = DURATION_PATTERN.search(rest)
    if dur_match:
        duration_raw = dur_match.group(1).strip()
        # Remove duration substring from rest
        rest = (rest[:dur_match.start()] + " " + rest[dur_match.end():]).strip()

    # 2. Extract Strength
    strength: str | None = None
    name_as_written: str = ""
    rest_after_strength: str = ""

    str_match = STRENGTH_PATTERN.search(rest)
    if str_match:
        strength = str_match.group(1).strip()
        name_as_written = rest[:str_match.start()].strip()
        rest_after_strength = rest[str_match.end():].strip()
    else:
        # Strength missing or unparseable
        tokens = rest.split()
        if tokens:
            name_as_written = tokens[0]
            rest_after_strength = " ".join(tokens[1:]).strip()

    # 3. Extract Frequency from rest_after_strength
    frequency_raw: str | None = None
    frequency_plain: str | None = None
    rest_after_freq: str = ""

    # Check dash pattern first
    dash_m = DASH_FREQ_PATTERN.search(rest_after_strength)
    known_m = KNOWN_FREQ_PATTERN.search(rest_after_strength)

    if dash_m:
        frequency_raw = dash_m.group(1).strip()
        rest_after_freq = (
            rest_after_strength[:dash_m.start()] + " " + rest_after_strength[dash_m.end():]
        ).strip()
    elif known_m:
        frequency_raw = known_m.group(1).strip()
        rest_after_freq = (
            rest_after_strength[:known_m.start()] + " " + rest_after_strength[known_m.end():]
        ).strip()
    else:
        # Check if first token could be an unknown frequency code
        tokens = rest_after_strength.split()
        if tokens:
            first_tok = tokens[0]
            first_tok_lower = first_tok.lower()
            known_instructions = terms.get("instructions", {})
            if first_tok_lower not in known_instructions and first_tok_lower not in ("before", "after", "with", "on", "for", "at"):
                cand_m = CANDIDATE_FREQ_TOKEN.match(first_tok)
                if cand_m:
                    frequency_raw = cand_m.group(1).strip()
                    rest_after_freq = " ".join(tokens[1:]).strip()
                else:
                    rest_after_freq = rest_after_strength
            else:
                rest_after_freq = rest_after_strength
        else:
            rest_after_freq = rest_after_strength

    if frequency_raw:
        frequency_plain = format_frequency_plain(frequency_raw, terms.get("frequencies", {}))

    # 4. Extract Instructions
    instructions_raw: str | None = None
    instructions_plain: str | None = None
    if rest_after_freq:
        instructions_raw = re.sub(r"\s+", " ", rest_after_freq).strip()
        if instructions_raw:
            inst_table = terms.get("instructions", {})
            instructions_plain = inst_table.get(instructions_raw.lower())

    # Assess if line was fully parsed or garbled
    warning: str | None = None
    is_garbled = (
        not name_as_written
        or not re.search(r"[a-zA-Z]{2,}", name_as_written)
        or strength is None
        or frequency_raw is None
        or (instructions_raw is not None and bool(re.search(r"[\?\*\{\}\<\>]{2,}", instructions_raw)))
    )

    if is_garbled:
        warning = f"Some details on this line could not be read: {mask_pii(source_snippet)}. Please check with your pharmacist."

    medication = Medication(
        name_as_written=mask_pii(name_as_written),
        form=mask_pii(form_as_written) if form_as_written else None,
        strength=mask_pii(strength) if strength else None,
        frequency_raw=mask_pii(frequency_raw) if frequency_raw else None,
        frequency_plain=mask_pii(frequency_plain) if frequency_plain else None,
        duration_raw=mask_pii(duration_raw) if duration_raw else None,
        instructions_raw=mask_pii(instructions_raw) if instructions_raw else None,
        instructions_plain=mask_pii(instructions_plain) if instructions_plain else None,
        source_snippet=mask_pii(source_snippet),
    )

    return medication, warning


def parse_prescription_text(text: str) -> PrescriptionSummary:
    """Parse raw prescription text deterministically into a PrescriptionSummary."""
    terms = load_prescription_terms()
    disclaimer = terms.get(
        "disclaimer",
        "This is a summary of what is written on your prescription. It is not a substitute for your doctor's or pharmacist's instructions. Do not change how you take any medicine without asking your doctor or pharmacist.",
    )

    if not text or not isinstance(text, str) or not text.strip():
        return PrescriptionSummary(
            medications=[],
            other_lines=[],
            warnings=[],
            disclaimer=disclaimer,
        )

    medications: list[Medication] = []
    other_lines: list[str] = []
    warnings: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # A line starting with an optional list number and a form word is always a medication line
        if FORM_PATTERN.search(line):
            med, warn = _parse_medication_line(line, terms)
            medications.append(med)
            if warn:
                warnings.append(warn)
        else:
            # Check if non-medication line should be dropped completely
            if _should_drop_line(line):
                continue
            # Non-medication line goes to other_lines with PII masked
            other_lines.append(mask_pii(line))

    return PrescriptionSummary(
        medications=medications,
        other_lines=other_lines,
        warnings=warnings,
        disclaimer=disclaimer,
    )


def parse_prescription_pdf(path: str) -> PrescriptionSummary:
    """Extract and parse prescription from digital PDF or flag image-only scanned document."""
    import pdfplumber

    terms = load_prescription_terms()
    disclaimer = terms.get(
        "disclaimer",
        "This is a summary of what is written on your prescription. It is not a substitute for your doctor's or pharmacist's instructions. Do not change how you take any medicine without asking your doctor or pharmacist.",
    )

    text_content = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text_content += extracted + "\n"

    if not text_content.strip():
        return PrescriptionSummary(
            medications=[],
            other_lines=[],
            warnings=["No text layer found. OCR is required for scanned documents."],
            disclaimer=disclaimer,
        )

    return parse_prescription_text(text_content)


def main() -> None:
    """CLI to parse a prescription PDF and print a readable summary."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse and summarize a prescription PDF deterministically."
    )
    parser.add_argument("pdf", help="Path to prescription PDF file")
    args = parser.parse_args()

    summary = parse_prescription_pdf(args.pdf)

    print("Prescription Summary:")
    print("=" * 60)

    if summary.medications:
        print("Medications:")
        print("-" * 60)
        for i, med in enumerate(summary.medications, 1):
            freq_display = med.frequency_raw or "None"
            if med.frequency_plain:
                freq_display = f"{med.frequency_raw} ({med.frequency_plain})"

            print(f"{i}. Name: {med.name_as_written}")
            if med.form:
                print(f"   Form: {med.form}")
            if med.strength:
                print(f"   Strength: {med.strength}")
            print(f"   Frequency: {freq_display}")
            if med.duration_raw:
                print(f"   Duration: {med.duration_raw}")
            if med.instructions_raw:
                inst_display = med.instructions_raw
                if med.instructions_plain:
                    inst_display = f"{med.instructions_raw} ({med.instructions_plain})"
                print(f"   Instructions: {inst_display}")
            print(f"   Source: {med.source_snippet}")
            print()
    else:
        print("No medications found.")
        print()

    if summary.other_lines:
        print("Other Lines:")
        print("-" * 60)
        for line in summary.other_lines:
            print(f"- {line}")
        print()

    if summary.warnings:
        print("Warnings:")
        print("-" * 60)
        for warn in summary.warnings:
            print(f"! {warn}")
        print()

    print("Disclaimer:")
    print("-" * 60)
    print(summary.disclaimer)


if __name__ == "__main__":
    main()
