"""Tests for deterministic prescription parsing and summarization service."""

from pathlib import Path

from app.services.prescription import parse_prescription_pdf, parse_prescription_text

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
TEXT_PDF_PATH = SAMPLES_DIR / "prescription_text.pdf"
SCANNED_PDF_PATH = SAMPLES_DIR / "prescription_scanned.pdf"

FORBIDDEN_WORDS = [
    "interaction",
    "side effect",
    "should",
    "recommend",
    "safe",
    "dangerous",
]

EXPECTED_DISCLAIMER = (
    "This is a summary of what is written on your prescription. "
    "It is not a substitute for your doctor's or pharmacist's instructions. "
    "Do not change how you take any medicine without asking your doctor or pharmacist."
)


def test_sample_prescription_text_parses_four_medications() -> None:
    """The sample parses into exactly 4 medications in order with exact fields."""
    summary = parse_prescription_pdf(str(TEXT_PDF_PATH))
    assert len(summary.medications) == 4

    # 1. Metformin, Tab, 500 mg, "1-0-1", "after food", "x 30 days"
    m1 = summary.medications[0]
    assert (
        m1.name_as_written,
        m1.form,
        m1.strength,
        m1.frequency_raw,
        m1.instructions_raw,
        m1.duration_raw,
    ) == ("Metformin", "Tab", "500 mg", "1-0-1", "after food", "x 30 days")

    # 2. Atorvastatin, Tab, 10 mg, "0-0-1", "at bedtime", "x 30 days"
    m2 = summary.medications[1]
    assert (
        m2.name_as_written,
        m2.form,
        m2.strength,
        m2.frequency_raw,
        m2.instructions_raw,
        m2.duration_raw,
    ) == ("Atorvastatin", "Tab", "10 mg", "0-0-1", "at bedtime", "x 30 days")

    # 3. Omeprazole, Cap, 20 mg, "OD", "before food", "x 14 days"
    m3 = summary.medications[2]
    assert (
        m3.name_as_written,
        m3.form,
        m3.strength,
        m3.frequency_raw,
        m3.instructions_raw,
        m3.duration_raw,
    ) == ("Omeprazole", "Cap", "20 mg", "OD", "before food", "x 14 days")

    # 4. Paracetamol, Syp, 650 mg, "SOS", "for fever"
    m4 = summary.medications[3]
    assert (
        m4.name_as_written,
        m4.form,
        m4.strength,
        m4.frequency_raw,
        m4.instructions_raw,
    ) == ("Paracetamol", "Syp", "650 mg", "SOS", "for fever")
    assert m4.duration_raw is None


def test_sample_frequency_plain_translations() -> None:
    """Frequency abbreviations and patterns map correctly to plain words."""
    summary = parse_prescription_pdf(str(TEXT_PDF_PATH))
    assert summary.medications[0].frequency_plain == "1 in the morning and 1 at night"
    assert summary.medications[1].frequency_plain == "1 at night"
    assert summary.medications[2].frequency_plain == "once a day"
    assert summary.medications[3].frequency_plain == "only when needed"


def test_other_lines_extracted_and_pii_dropped() -> None:
    """Advice and notes are in other_lines, while patient and doctor PII is completely dropped."""
    summary = parse_prescription_pdf(str(TEXT_PDF_PATH))

    assert "Avoid sugary drinks" in summary.other_lines
    assert "Review after 2 weeks with HbA1c report" in summary.other_lines

    # Check serialized summary
    serialized = summary.model_dump_json()
    assert "TEST PATIENT" not in serialized
    assert "Sample Doctor" not in serialized
    assert "9876543210" not in serialized


def test_fractions_unknown_abbreviations_and_garbled_lines() -> None:
    """Fractions translate to plain words, unknown codes stay raw, and garbled lines add warnings."""
    # Fraction frequency pattern
    fraction_summary = parse_prescription_text("Tab Amlodipine 5 mg 1/2-0-1/2")
    assert len(fraction_summary.medications) == 1
    med_frac = fraction_summary.medications[0]
    assert med_frac.name_as_written == "Amlodipine"
    assert med_frac.strength == "5 mg"
    assert med_frac.frequency_raw == "1/2-0-1/2"
    assert med_frac.frequency_plain == "1/2 in the morning and 1/2 at night"

    # Unknown abbreviation keeps raw and leaves plain as None
    unknown_summary = parse_prescription_text("Tab Aspirin 75 mg UNK after food")
    assert len(unknown_summary.medications) == 1
    med_unk = unknown_summary.medications[0]
    assert med_unk.frequency_raw == "UNK"
    assert med_unk.frequency_plain is None
    assert med_unk.instructions_raw == "after food"

    # Garbled line keeps snippet and yields warning
    garbled_summary = parse_prescription_text("Tab GarbledMedicine ???")
    assert len(garbled_summary.medications) == 1
    med_garbled = garbled_summary.medications[0]
    assert med_garbled.source_snippet == "Tab GarbledMedicine ???"
    assert len(garbled_summary.warnings) >= 1
    assert "Some details on this line could not be read: Tab GarbledMedicine ???. Please check with your pharmacist." in garbled_summary.warnings


def test_scanned_prescription_returns_no_medications_and_ocr_warning() -> None:
    """Scanned image-only PDF returns empty medications and OCR warning."""
    summary = parse_prescription_pdf(str(SCANNED_PDF_PATH))
    assert len(summary.medications) == 0
    assert "No text layer found. OCR is required for scanned documents." in summary.warnings


def test_serialized_summary_forbidden_words_and_disclaimer() -> None:
    """Serialized summary contains none of the forbidden words and contains the fixed disclaimer."""
    summary_text = parse_prescription_pdf(str(TEXT_PDF_PATH))
    summary_scanned = parse_prescription_pdf(str(SCANNED_PDF_PATH))

    for summary in [summary_text, summary_scanned]:
        serialized = summary.model_dump_json().lower()
        for forbidden in FORBIDDEN_WORDS:
            assert forbidden not in serialized, f"Forbidden word '{forbidden}' found in summary"
        assert summary.disclaimer == EXPECTED_DISCLAIMER


def test_parse_prescription_text_handles_empty_and_garbage_input() -> None:
    """parse_prescription_text raises no exception on empty string or random garbage text."""
    res_empty = parse_prescription_text("")
    assert len(res_empty.medications) == 0
    assert len(res_empty.other_lines) == 0
    assert res_empty.disclaimer == EXPECTED_DISCLAIMER

    res_spaces = parse_prescription_text("   \n   \t  \n  ")
    assert len(res_spaces.medications) == 0

    res_garbage = parse_prescription_text("~!@#$%^&*()_+ Random non-medication text without forms")
    assert len(res_garbage.medications) == 0
    assert len(res_garbage.other_lines) == 1
    assert res_garbage.other_lines[0] == "~!@#$%^&*()_+ Random non-medication text without forms"


def test_header_word_boundary_and_medication_line_rules() -> None:
    """Drops Moxifloxacin is parsed with form Drops, Idoxuridine/Agomelatine not dropped, headers dropped."""
    text = (
        "Drops Moxifloxacin 0.5% 1 drop BD x 5 days\n"
        "Idoxuridine ointment\n"
        "Agomelatine 25 mg OD\n"
        "Dr. Sample Doctor\n"
        "ID: 12345\n"
        "Age: 45\n"
        "Name: X\n"
        "Ph: 9876543210\n"
    )
    summary = parse_prescription_text(text)

    # 1. "Drops Moxifloxacin 0.5% 1 drop BD x 5 days" is parsed as a medication with form Drops
    assert len(summary.medications) == 1
    med = summary.medications[0]
    assert med.name_as_written == "Moxifloxacin"
    assert med.form == "Drops"
    assert med.strength == "0.5%"
    assert med.frequency_raw == "BD"
    assert med.duration_raw == "x 5 days"

    # 2. "Idoxuridine ointment" and "Agomelatine 25 mg OD" are NOT dropped as header lines
    assert "Idoxuridine ointment" in summary.other_lines
    assert "Agomelatine 25 mg OD" in summary.other_lines

    # 3. "Dr. Sample Doctor", "ID: 12345", "Age: 45", "Name: X" and "Ph: 9876543210" are still dropped everywhere
    serialized = summary.model_dump_json()
    assert "Dr. Sample Doctor" not in serialized
    assert "ID: 12345" not in serialized
    assert "Age: 45" not in serialized
    assert "Name: X" not in serialized
    assert "9876543210" not in serialized

