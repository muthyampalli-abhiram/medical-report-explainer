"""Script to verify live LLM explanation wording quality, readability, caching, and guardrails."""

from datetime import date
from pathlib import Path
import re
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import settings
from app.data.catalog import load_test_catalog
from app.models.lab import ExtractionResponse, LabReport, LabResult, ResultFlag
from app.services.explainer import BANNED_WORDS_RE, clear_llm_cache, explain_report
from app.services.llm_client import get_llm_client
from scripts.evaluate_all import compute_flesch_kincaid_grade


def main() -> None:
    if (settings.LLM_PROVIDER or "").lower().strip() != "groq" or not settings.LLM_API_KEY:
        print("skipped")
        sys.exit(0)

    catalog = load_test_catalog()
    results = []
    for entry in catalog:
        canon = entry["canonical_name"]
        val = 90.0 if canon != "Urine Protein" else None
        val_txt = "Negative" if canon == "Urine Protein" else None
        unit = (
            "%"
            if canon == "HbA1c"
            else (
                "mmol/L"
                if canon == "Potassium"
                else ("g/dL" if canon == "Hemoglobin" else "mg/dL")
            )
        )
        if canon == "Urine Protein":
            unit = None

        results.append(
            LabResult(
                test_name_raw=canon,
                test_name_canonical=canon,
                value=val,
                value_text=val_txt,
                unit=unit,
                flag=ResultFlag.NORMAL,
            )
        )

    dummy_resp = ExtractionResponse(
        report=LabReport(
            report_date=date(2026, 1, 15),
            lab_name="Live Wording Check Lab",
            results=results,
        ),
        warnings=[],
    )

    client = get_llm_client()
    runs_explanations = []
    all_grades: list[float] = []
    fallback_count = 0
    total_outputs = 0
    banned_detected: list[str] = []

    print("=" * 70)
    print("LIVE LLM EXPLANATION WORDING QUALITY CHECK")
    print(f"Provider: {settings.LLM_PROVIDER} | Model: {settings.LLM_MODEL}")
    print("=" * 70)

    # 3 runs with cache cleared before each run
    for run_idx in range(1, 4):
        clear_llm_cache()
        print(f"\n--- Run {run_idx} (Cache Cleared) ---")
        expl = explain_report(dummy_resp, client, language="en")
        runs_explanations.append(expl)

        for e in expl.explanations:
            total_outputs += 1
            if e.used_fallback:
                fallback_count += 1
            grade = compute_flesch_kincaid_grade(e.what_it_measures)
            all_grades.append(grade)

            clean_text = re.sub(r"\b(low|high)[-\s]density\b", "", e.what_it_measures, flags=re.IGNORECASE)
            if BANNED_WORDS_RE.search(clean_text):
                banned_detected.append(f"[{e.test_name} Run {run_idx}]: {e.what_it_measures}")

            print(f"Test: {e.test_name}")
            print(f"  Fallback: {e.used_fallback}")
            print(f"  FK Grade: {grade}")
            print(f"  Text: {e.what_it_measures}")

    # 1 extra run WITHOUT clearing cache to verify identical output
    print(f"\n--- Run 4 (Cached from Run 3, Cache NOT Cleared) ---")
    expl4 = explain_report(dummy_resp, client, language="en")

    run3_texts = [e.what_it_measures for e in runs_explanations[2].explanations]
    run4_texts = [e.what_it_measures for e in expl4.explanations]
    cached_identical = (run3_texts == run4_texts)

    for e in expl4.explanations:
        grade = compute_flesch_kincaid_grade(e.what_it_measures)
        print(f"Test: {e.test_name}")
        print(f"  Fallback: {e.used_fallback}")
        print(f"  FK Grade: {grade}")
        print(f"  Text: {e.what_it_measures}")

    mean_grade = sum(all_grades) / len(all_grades) if all_grades else 0.0
    max_grade = max(all_grades) if all_grades else 0.0

    print("\n" + "=" * 70)
    print("LIVE WORDING CHECK SUMMARY")
    print("=" * 70)
    print(f"Number of outputs evaluated (Runs 1-3): {total_outputs}")
    print(f"Fallbacks triggered: {fallback_count}")
    print(f"Mean Flesch-Kincaid Grade: {mean_grade:.2f}")
    print(f"Maximum Flesch-Kincaid Grade: {max_grade:.2f}")
    print(f"Cached run identical to previous run: {cached_identical}")
    if banned_detected:
        print(f"Outputs containing banned words ({len(banned_detected)}):")
        for item in banned_detected:
            print(f"  - {item}")
    else:
        print("Outputs containing banned words: None (0 detected)")
    print("=" * 70)


if __name__ == "__main__":
    main()
