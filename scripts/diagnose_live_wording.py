"""Script to diagnose live LLM explanation failures and reason codes."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import settings
from app.services.explainer import LAST_LLM_DIAGNOSTICS, clear_llm_cache, explain_report
from app.services.llm_client import get_llm_client
from app.services.pdf_extractor import extract_report_from_pdf
from app.services.pipeline import process_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose live LLM wording and rejection reasons.")
    parser.add_argument("--model", default=None, help="Override LLM_MODEL for this run")
    args = parser.parse_args()

    if (settings.LLM_PROVIDER or "").lower().strip() != "groq" or not settings.LLM_API_KEY:
        print("skipped")
        sys.exit(0)

    if args.model:
        settings.LLM_MODEL = args.model

    pdf_path = ROOT_DIR / "samples" / "report_table.pdf"
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        sys.exit(1)

    extraction = extract_report_from_pdf(str(pdf_path))
    processed = process_report(extraction)

    expected_tests = list(
        dict.fromkeys(
            r.test_name_canonical for r in processed.report.results if r.test_name_canonical
        )
    )

    client = get_llm_client()
    all_reasons: list[str] = []

    print("=" * 70)
    print("LIVE LLM WORDING DIAGNOSIS")
    print(f"Provider: {settings.LLM_PROVIDER} | Model: {settings.LLM_MODEL}")
    print(f"Sample PDF: {pdf_path.name} ({len(expected_tests)} explainable tests)")
    print("=" * 70)

    for run_idx in range(1, 4):
        clear_llm_cache()
        print(f"\n--- Run {run_idx} ---")

        raised_exc_name = "None"
        try:
            explain_report(processed, client, language="en")
        except Exception as exc:
            raised_exc_name = exc.__class__.__name__

        status_code = getattr(client, "last_status_code", None)
        finish_reason = getattr(client, "last_finish_reason", None)
        completion_tokens = getattr(client, "last_completion_tokens", None)
        raw_content = getattr(client, "last_raw_content", None) or ""
        raw_snippet = raw_content[:1500]

        parsed_json_ok = False
        present_tests: list[str] = []
        if raw_content:
            try:
                parsed_data = json.loads(raw_content)
                parsed_json_ok = True
                if isinstance(parsed_data, dict) and isinstance(parsed_data.get("tests"), list):
                    for item in parsed_data["tests"]:
                        if isinstance(item, dict) and "name" in item:
                            present_tests.append(str(item["name"]))
            except Exception:
                parsed_json_ok = False

        missing_tests = [t for t in expected_tests if t not in present_tests]

        print(f"Call raised: {raised_exc_name}")
        print(f"last_status_code: {status_code}")
        print(f"last_finish_reason: {finish_reason}")
        print(f"last_completion_tokens: {completion_tokens}")
        print(f"parsed_as_json: {parsed_json_ok}")
        print(f"Present test names ({len(present_tests)}): {present_tests}")
        print(f"Missing test names ({len(missing_tests)}): {missing_tests}")
        print(f"Raw content snippet (first 1500 chars):\n{raw_snippet}")
        print("\nPer-test reason codes:")
        for diag in LAST_LLM_DIAGNOSTICS:
            print(f"  - {diag['test']}: {diag['reason']}")
            all_reasons.append(diag["reason"])

        if run_idx < 3:
            print("\nWaiting 20 seconds before next run...")
            time.sleep(20)

    print("\n" + "=" * 70)
    print("DIAGNOSIS TALLY")
    print("=" * 70)
    tally = Counter(all_reasons)
    for reason, count in tally.most_common():
        print(f"  {reason}: {count}")
    print("=" * 70)


if __name__ == "__main__":
    main()
