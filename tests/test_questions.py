"""Tests for deterministic doctor question generator service."""

from datetime import date
from pathlib import Path

from app.models.lab import ExtractionResponse, LabReport, LabResult, ResultFlag
from app.models.trend import TrendPoint
from app.services.pdf_extractor import extract_report_from_pdf
from app.services.pipeline import process_report
from app.services.questions import generate_questions, trend_note
from app.services.safety import check_text

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
BANNED_WORDS = [
    "improved",
    "worsened",
    "better",
    "worse",
    "good",
    "bad",
    "concerning",
    "reassuring",
]


def test_hba1c_high_question_on_sample_table() -> None:
    """HbA1c HIGH on report_table.pdf produces expected question string."""
    pdf_path = SAMPLES_DIR / "report_table.pdf"
    raw_resp = extract_report_from_pdf(str(pdf_path))
    processed_resp = process_report(raw_resp)

    dq = generate_questions(processed_resp)
    hba1c_q = next((q for q in dq.questions if q.related_test == "HbA1c"), None)
    assert hba1c_q is not None

    expected = (
        "My HbA1c result was 7.4 %, which is above the reference range of 4.0 - 5.6 %. "
        "What could be contributing to this, and does it need any follow-up?"
    )
    assert hba1c_q.text == expected


def test_normal_results_produce_no_questions() -> None:
    """NORMAL results (HDL Cholesterol, Hemoglobin, Urine Protein) produce no questions."""
    pdf_path = SAMPLES_DIR / "report_table.pdf"
    raw_resp = extract_report_from_pdf(str(pdf_path))
    processed_resp = process_report(raw_resp)

    dq = generate_questions(processed_resp)
    question_tests = {q.related_test for q in dq.questions}

    assert "HDL Cholesterol" not in question_tests
    assert "Hemoglobin" not in question_tests
    assert "Urine Protein" not in question_tests


def test_critical_potassium_comes_first() -> None:
    """Critical potassium 7.0 gives the urgent question and comes first in priority."""
    report = LabReport(
        results=[
            LabResult(
                test_name_raw="HbA1c",
                test_name_canonical="HbA1c",
                value=7.4,
                unit="%",
                ref_text="4.0 - 5.6",
                ref_low=4.0,
                ref_high=5.6,
                flag=ResultFlag.HIGH,
            ),
            LabResult(
                test_name_raw="Potassium",
                test_name_canonical="Potassium",
                value=7.0,
                unit="mmol/L",
                ref_text="3.5 - 5.0",
                ref_low=3.5,
                ref_high=5.0,
                flag=ResultFlag.CRITICAL_HIGH,
            ),
            LabResult(
                test_name_raw="Fasting Blood Sugar",
                test_name_canonical="Fasting Glucose",
                value=132.0,
                unit="mg/dL",
                ref_text="70 - 99",
                ref_low=70.0,
                ref_high=99.0,
                flag=ResultFlag.HIGH,
            ),
        ]
    )
    resp = ExtractionResponse(report=report)
    dq = generate_questions(resp)

    assert len(dq.questions) == 3
    first_q = dq.questions[0]
    assert first_q.related_test == "Potassium"
    expected = (
        "My Potassium result was 7.0 mmol/L, which is far outside the usual range. "
        "Do I need to be seen urgently?"
    )
    assert first_q.text == expected


def test_eight_question_cap_and_dropped_trends() -> None:
    """The 8-question cap works with 12 abnormal results, and dropped trends appear in trend_notes."""
    results = [
        LabResult(
            test_name_raw=f"Test_{i}",
            test_name_canonical=f"Test_{i}",
            value=float(100 + i),
            unit="mg/dL",
            ref_text="50 - 90",
            ref_low=50.0,
            ref_high=90.0,
            flag=ResultFlag.HIGH,
        )
        for i in range(12)
    ]
    report = LabReport(results=results)
    resp = ExtractionResponse(report=report)

    trends = {
        "HbA1c": [
            TrendPoint(
                report_id="rep1",
                report_date=date(2026, 1, 15),
                value=7.4,
                unit="%",
                flag=ResultFlag.HIGH,
            ),
            TrendPoint(
                report_id="rep2",
                report_date=date(2026, 4, 20),
                value=6.9,
                unit="%",
                flag=ResultFlag.HIGH,
            ),
        ]
    }

    dq = generate_questions(resp, trends=trends)

    # Capped at 8 questions total
    assert len(dq.questions) == 8

    # Dropped trend should appear in trend_notes
    assert len(dq.trend_notes) >= 1
    assert any("HbA1c" in note and "decreased" in note for note in dq.trend_notes)


def test_four_fixed_general_questions_always_present() -> None:
    """The 4 fixed general questions are always present."""
    report = LabReport(results=[])
    resp = ExtractionResponse(report=report)
    dq = generate_questions(resp)

    expected_general = [
        "Are there any other tests I should have?",
        "When should I repeat these tests?",
        "Is there anything in my diet, activity, or daily routine I should discuss with you?",
        "Should I bring any of these results to my next appointment with another specialist?",
    ]
    assert dq.general_questions == expected_general


def test_trend_note_neutral_phrasing_and_rules() -> None:
    """trend_note handles decrease, flat trends, single points, and contains no banned words."""
    pts_decrease = [
        TrendPoint(
            report_id="r1",
            report_date=date(2026, 1, 15),
            value=7.4,
            unit="%",
            flag=ResultFlag.HIGH,
        ),
        TrendPoint(
            report_id="r2",
            report_date=date(2026, 4, 20),
            value=6.9,
            unit="%",
            flag=ResultFlag.HIGH,
        ),
    ]

    note_decrease = trend_note(pts_decrease, test_name="HbA1c")
    assert note_decrease is not None
    assert "decreased" in note_decrease
    assert "7.4" in note_decrease
    assert "6.9" in note_decrease
    for banned in BANNED_WORDS:
        assert banned not in note_decrease.lower()

    # Flat trend (< 2% change)
    pts_flat = [
        TrendPoint(
            report_id="r1",
            report_date=date(2026, 1, 15),
            value=100.0,
            unit="mg/dL",
            flag=ResultFlag.NORMAL,
        ),
        TrendPoint(
            report_id="r2",
            report_date=date(2026, 4, 20),
            value=101.0,
            unit="mg/dL",
            flag=ResultFlag.NORMAL,
        ),
    ]
    note_flat = trend_note(pts_flat, test_name="Glucose")
    assert note_flat is not None
    assert "stayed about the same" in note_flat

    # Single point returns None
    pts_single = [pts_decrease[0]]
    assert trend_note(pts_single, test_name="HbA1c") is None


def test_every_generated_string_passes_safety_check() -> None:
    """Every generated question, general question, and trend note passes safety.check_text."""
    pdf_path = SAMPLES_DIR / "report_table.pdf"
    raw_resp = extract_report_from_pdf(str(pdf_path))
    processed_resp = process_report(raw_resp)

    trends = {
        "HbA1c": [
            TrendPoint(
                report_id="r1",
                report_date=date(2026, 1, 15),
                value=7.4,
                unit="%",
                flag=ResultFlag.HIGH,
            ),
            TrendPoint(
                report_id="r2",
                report_date=date(2026, 4, 20),
                value=6.9,
                unit="%",
                flag=ResultFlag.HIGH,
            ),
        ]
    }

    dq = generate_questions(processed_resp, trends=trends)

    for q in dq.questions:
        assert check_text(q.text) == [], f"Question violated safety: {q.text}"
    for gq in dq.general_questions:
        assert check_text(gq) == [], f"General question violated safety: {gq}"
    for tn in dq.trend_notes:
        assert check_text(tn) == [], f"Trend note violated safety: {tn}"
