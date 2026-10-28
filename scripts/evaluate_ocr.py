"""Evaluation script comparing OCR results against expected JSON key."""

import json
from pathlib import Path
import sys

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from app.models.lab import LabResult
from app.services.ocr_extractor import extract_report_from_image
from app.services.ocr_safeguards import apply_ocr_safeguards, filter_ocr_junk
from app.services.pipeline import process_report

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
EXPECTED_PATH = SAMPLES_DIR / "report_table.expected.json"


def evaluate_image(image_path: Path, expected: dict) -> dict:
    """Evaluate OCR on a single image against expected report data."""
    raw_response = extract_report_from_image(image_path)
    processed = process_report(raw_response)
    processed = filter_ocr_junk(processed)
    processed = apply_ocr_safeguards(processed)

    results: list[LabResult] = processed.report.results
    expected_rows: list[dict] = expected["results"]

    rows_correct = 0
    rows_wrong = 0
    fields_correct = 0
    total_fields = len(expected_rows) * 4  # name, value, unit, range

    matched_indices = set()

    for exp in expected_rows:
        exp_name = exp["test_name_raw"]
        exp_val = exp.get("value")
        exp_val_text = exp.get("value_text")
        exp_unit = exp.get("unit")
        exp_low = exp.get("ref_low")
        exp_high = exp.get("ref_high")

        # Find matching extracted result
        match_idx = None
        for idx, act in enumerate(results):
            if idx in matched_indices:
                continue
            if (
                act.test_name_raw == exp_name
                or act.test_name_canonical == exp_name
                or (act.test_name_canonical and exp_name.lower() in act.test_name_raw.lower())
            ):
                match_idx = idx
                break

        if match_idx is not None:
            matched_indices.add(match_idx)
            act = results[match_idx]

            # Check individual fields
            name_ok = (act.test_name_raw == exp_name) or (act.test_name_canonical == exp_name)
            val_ok = (
                (act.value is not None and exp_val is not None and abs(act.value - exp_val) < 1e-4)
                if exp_val is not None
                else (act.value_text == exp_val_text)
            )
            unit_ok = act.unit == exp_unit
            ref_ok = (act.ref_low == exp_low) and (act.ref_high == exp_high)

            row_fields_correct = sum([name_ok, val_ok, unit_ok, ref_ok])
            fields_correct += row_fields_correct

            if row_fields_correct == 4:
                rows_correct += 1
            else:
                rows_wrong += 1

    # Any extracted result not matching expected
    extra_results = len(results) - len(matched_indices)
    rows_wrong += extra_results

    rows_missing = len(expected_rows) - (rows_correct + (rows_wrong - extra_results))

    avg_conf = (
        sum(r.confidence for r in results) / len(results)
        if results
        else 0.0
    )

    return {
        "image": image_path.name,
        "rows_correct": rows_correct,
        "rows_wrong": rows_wrong,
        "rows_missing": rows_missing,
        "avg_confidence": avg_conf,
        "warnings": processed.warnings,
        "fields_correct": fields_correct,
        "total_fields": total_fields,
    }


def main():
    with open(EXPECTED_PATH, "r", encoding="utf-8") as f:
        expected = json.load(f)

    images = [
        SAMPLES_DIR / "photo_clean.png",
        SAMPLES_DIR / "photo_skewed.jpg",
        SAMPLES_DIR / "photo_poor.jpg",
    ]

    total_fields_correct = 0
    total_fields_count = 0

    print("================ OCR PIPELINE EVALUATION ================\n")

    for img_path in images:
        if not img_path.exists():
            print(f"Skipping {img_path.name} (not found)")
            continue

        stats = evaluate_image(img_path, expected)
        total_fields_correct += stats["fields_correct"]
        total_fields_count += stats["total_fields"]

        print(f"Image: {stats['image']}")
        print(f"  Rows Exactly Correct: {stats['rows_correct']} / {len(expected['results'])}")
        print(f"  Rows Wrong:           {stats['rows_wrong']}")
        print(f"  Rows Missing:         {stats['rows_missing']}")
        print(f"  Average Confidence:   {stats['avg_confidence']:.2%}")
        if stats["warnings"]:
            print(f"  Warnings ({len(stats['warnings'])}):")
            for w in stats["warnings"]:
                print(f"    - {w}")
        else:
            print("  Warnings:             None")
        print()

    overall_accuracy = (
        (total_fields_correct / total_fields_count) * 100
        if total_fields_count > 0
        else 0.0
    )
    print("=========================================================")
    print(f"Overall Field-Level Accuracy: {overall_accuracy:.2f}% ({total_fields_correct}/{total_fields_count} fields)")
    print("=========================================================")


if __name__ == "__main__":
    main()
