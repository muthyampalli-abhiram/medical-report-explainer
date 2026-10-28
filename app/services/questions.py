"""Deterministic doctor question generation service based on lab results and biomarker trends."""

import json
from pathlib import Path
import re
from typing import Any

from app.models.lab import ExtractionResponse, LabResult, ResultFlag
from app.models.questions import DoctorQuestion, DoctorQuestions
from app.models.trend import TrendPoint
from app.services.safety import check_text

BANNED_TREND_WORDS = {
    "improved",
    "worsened",
    "better",
    "worse",
    "good",
    "bad",
    "concerning",
    "reassuring",
}


def load_question_templates() -> dict[str, Any]:
    """Load question templates from the package data directory."""
    templates_path = Path(__file__).resolve().parent.parent / "data" / "question_templates.json"
    with open(templates_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clean_text(text: str) -> str:
    """Normalize whitespace and fix spacing before punctuation."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.:;?!])", r"\1", text)
    return text.strip()


def _format_range(result: LabResult) -> str:
    """Format reference range string cleanly with unit if needed."""
    unit = result.unit or ""
    if result.ref_text and result.ref_text.strip():
        ref_txt = result.ref_text.strip()
        if unit and unit not in ref_txt:
            return f"{ref_txt} {unit}".strip()
        return ref_txt

    if result.ref_low is not None and result.ref_high is not None:
        low_str = f"{result.ref_low:g}"
        high_str = f"{result.ref_high:g}"
        return f"{low_str} - {high_str} {unit}".strip()
    if result.ref_high is not None:
        high_str = f"{result.ref_high:g}"
        return f"< {high_str} {unit}".strip()
    if result.ref_low is not None:
        low_str = f"{result.ref_low:g}"
        return f"> {low_str} {unit}".strip()
    return ""


def trend_note(points: list[TrendPoint], test_name: str | None = None) -> str | None:
    """Generate a strictly neutral observation note for biomarker trends over time.

    Needs at least 2 points.
    Uses ONLY the words increased, decreased, stayed about the same, or changed.
    NEVER uses improved, worsened, better, worse, good, bad, concerning, or reassuring.
    """
    if not points or len(points) < 2:
        return None

    sorted_pts = sorted(points, key=lambda p: p.report_date)
    first_pt = sorted_pts[0]
    last_pt = sorted_pts[-1]

    name = test_name or "Result"
    date1 = first_pt.report_date.isoformat()
    date2 = last_pt.report_date.isoformat()
    unit = last_pt.unit or first_pt.unit or ""
    unit_str = f" {unit}" if unit else ""

    first_val = first_pt.value
    last_val = last_pt.value

    if first_val != 0:
        pct_change = abs(last_val - first_val) / abs(first_val)
    else:
        pct_change = 0.0 if last_val == 0 else 1.0

    if pct_change < 0.02:
        note = f"{name} stayed about the same ({first_val} to {last_val}) between {date1} and {date2}."
    else:
        verb = "increased" if last_val > first_val else "decreased"
        note = f"{name} {verb} from {first_val} to {last_val}{unit_str} between {date1} and {date2}."

    note_clean = _clean_text(note)

    # Verify no banned words are present
    words_in_note = set(re.findall(r"\b[a-zA-Z]+\b", note_clean.lower()))
    if words_in_note & BANNED_TREND_WORDS:
        return None

    # Safety check
    if check_text(note_clean):
        return None

    return note_clean


def generate_questions(
    response: ExtractionResponse,
    trends: dict[str, list[TrendPoint]] | None = None,
) -> DoctorQuestions:
    """Generate deterministic, patient-oriented questions for their doctor.

    Order:
      1. Critical results (CRITICAL_LOW, CRITICAL_HIGH)
      2. Abnormal results (HIGH, LOW)
      3. UNKNOWN reference range results (recognized tests with numeric values)
      4. Trend questions (tests with >= 2 points)

    Capped at 8 specific questions total.
    Dropped trends beyond the cap are retained as trend_notes.
    Every generated string must pass safety.check_text.
    """
    templates = load_question_templates()

    critical_candidates: list[DoctorQuestion] = []
    high_low_candidates: list[DoctorQuestion] = []
    unknown_candidates: list[DoctorQuestion] = []

    for result in response.report.results:
        # Only results with a non-None canonical test name and a numeric value get questions
        if result.test_name_canonical is None or result.value is None:
            continue

        # Normal results get no question
        if result.flag == ResultFlag.NORMAL:
            continue

        template: str | None = None
        reason: str = ""

        if result.flag in (ResultFlag.CRITICAL_LOW, ResultFlag.CRITICAL_HIGH):
            template = templates.get(result.flag.value) or templates.get("CRITICAL_HIGH")
            reason = f"Critical {result.flag.value.lower().replace('_', ' ')} result"
        elif result.flag == ResultFlag.HIGH:
            template = templates.get("HIGH")
            reason = "High result above reference range"
        elif result.flag == ResultFlag.LOW:
            template = templates.get("LOW")
            reason = "Low result below reference range"
        elif result.flag == ResultFlag.UNKNOWN:
            template = templates.get("UNKNOWN")
            reason = "Missing reference range"

        if not template:
            continue

        range_str = _format_range(result)
        unit_str = result.unit or ""
        q_text = _clean_text(
            template.format(
                test=result.test_name_canonical,
                value=result.value,
                unit=unit_str,
                range=range_str,
            )
        )

        if check_text(q_text):
            continue

        question = DoctorQuestion(
            text=q_text,
            related_test=result.test_name_canonical,
            reason=reason,
        )

        if result.flag in (ResultFlag.CRITICAL_LOW, ResultFlag.CRITICAL_HIGH):
            critical_candidates.append(question)
        elif result.flag in (ResultFlag.HIGH, ResultFlag.LOW):
            high_low_candidates.append(question)
        elif result.flag == ResultFlag.UNKNOWN:
            unknown_candidates.append(question)

    # Process trend candidates
    trend_candidates: list[tuple[DoctorQuestion, str, list[TrendPoint]]] = []
    if trends:
        trend_template = templates.get("trend")
        if trend_template:
            for test_name, points in trends.items():
                if not points or len(points) < 2:
                    continue
                sorted_pts = sorted(points, key=lambda p: p.report_date)
                first_pt = sorted_pts[0]
                last_pt = sorted_pts[-1]
                unit_str = last_pt.unit or first_pt.unit or ""
                t_text = _clean_text(
                    trend_template.format(
                        test=test_name,
                        first_value=first_pt.value,
                        unit=unit_str,
                        first_date=first_pt.report_date.isoformat(),
                        last_value=last_pt.value,
                        last_date=last_pt.report_date.isoformat(),
                    )
                )
                if check_text(t_text):
                    continue

                t_question = DoctorQuestion(
                    text=t_text,
                    related_test=test_name,
                    reason=f"Historical trend observation for {test_name}",
                )
                trend_candidates.append((t_question, test_name, sorted_pts))

    # Assemble up to 8 questions by priority
    final_questions: list[DoctorQuestion] = []
    dropped_trends: list[tuple[str, list[TrendPoint]]] = []

    for q in critical_candidates:
        if len(final_questions) < 8:
            final_questions.append(q)

    for q in high_low_candidates:
        if len(final_questions) < 8:
            final_questions.append(q)

    for q in unknown_candidates:
        if len(final_questions) < 8:
            final_questions.append(q)

    for q, test_name, pts in trend_candidates:
        if len(final_questions) < 8:
            final_questions.append(q)
        else:
            dropped_trends.append((test_name, pts))

    # Retain dropped trends as trend_notes
    trend_notes: list[str] = []
    for test_name, pts in dropped_trends:
        note = trend_note(pts, test_name=test_name)
        if note and not check_text(note):
            trend_notes.append(note)

    # General questions
    general_questions = [
        _clean_text(gq)
        for gq in templates.get("general_questions", [])
        if not check_text(gq)
    ]

    return DoctorQuestions(
        report_id=response.report.report_id,
        questions=final_questions,
        general_questions=general_questions,
        trend_notes=trend_notes,
    )


def main() -> None:
    """CLI to run extraction, pipeline, and question generation on a lab report PDF."""
    import argparse
    import sys
    import tempfile
    from app.db.database import init_db
    from app.services.pdf_extractor import extract_report_from_pdf
    from app.services.pipeline import process_report
    from app.services.repository import get_trend, list_profiles, save_report

    parser = argparse.ArgumentParser(
        description="Extract and generate doctor questions from a lab report PDF."
    )
    parser.add_argument("pdf", help="Path to lab report PDF file")
    parser.add_argument(
        "--with-demo-trend",
        action="store_true",
        help="Include demo biomarker trends from sample reports",
    )
    args = parser.parse_args()

    raw_response = extract_report_from_pdf(args.pdf)
    processed_response = process_report(raw_response)

    trends: dict[str, list[TrendPoint]] | None = None
    if args.with_demo_trend:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            temp_db = Path(tmpdir) / "demo_trend.db"
            init_db(temp_db)
            profiles = list_profiles(temp_db)
            profile_id = profiles[0]["id"]

            samples_dir = Path(__file__).resolve().parent.parent.parent / "samples"
            table_pdf = samples_dir / "report_table.pdf"
            lines_pdf = samples_dir / "report_lines.pdf"

            rep1 = process_report(extract_report_from_pdf(str(table_pdf)))
            rep2 = process_report(extract_report_from_pdf(str(lines_pdf)))
            save_report(profile_id, rep1, temp_db)
            save_report(profile_id, rep2, temp_db)

            # Also save input PDF if different content
            try:
                save_report(profile_id, processed_response, temp_db)
            except Exception:
                pass

            trends = {}
            for res in processed_response.report.results:
                if res.test_name_canonical:
                    pts = get_trend(profile_id, res.test_name_canonical, temp_db)
                    if len(pts) >= 2:
                        trends[res.test_name_canonical] = pts

    doc_questions = generate_questions(processed_response, trends=trends)

    print(f"Doctor Questions for Report {doc_questions.report_id}:")
    print("=" * 60)
    if doc_questions.questions:
        for i, q in enumerate(doc_questions.questions, 1):
            print(f"{i}. {q.text}")
            if q.reason:
                print(f"   [Reason: {q.reason}]")
    else:
        print("No specific biomarker questions generated.")

    print()
    print("General Questions:")
    print("-" * 60)
    for gq in doc_questions.general_questions:
        print(f"- {gq}")

    if doc_questions.trend_notes:
        print()
        print("Trend Notes:")
        print("-" * 60)
        for tn in doc_questions.trend_notes:
            print(f"- {tn}")


if __name__ == "__main__":
    main()
