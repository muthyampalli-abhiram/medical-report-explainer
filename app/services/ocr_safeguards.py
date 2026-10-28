import re
from typing import Any

from app.config import settings
from app.data.catalog import load_test_catalog
from app.models.lab import ExtractionResponse, ResultFlag

FORBIDDEN_EXACT = {
    "negative",
    "positive",
    "normal",
    "trace",
    "nil",
    "absent",
    "present",
    "reactive",
    "non reactive",
    "non-reactive",
}

FORBIDDEN_SUBSTRINGS = ("mg/dl", "g/dl", "mmol/l", "%", "/")


def filter_ocr_junk(response: ExtractionResponse) -> ExtractionResponse:
    """Filter out OCR junk lines from results before applying safeguards.

    Applied to OCR results only (after process_report, before apply_ocr_safeguards).
    For each result WITHOUT a canonical test name, keep it only if ALL are true:
      - confidence >= OCR_VERIFY_BELOW
      - raw name has at least 3 consecutive letters
      - contains no digits
      - contains none of the strings "mg/dl", "g/dl", "mmol/l", "%", "/"
      - its lowercase form is not one of: negative, positive, normal, trace, nil, absent, present, reactive, non reactive
    Otherwise remove it.
    Results WITH a canonical name are never removed by this filter.
    Never modifies a value.
    If any rows were removed, adds ONE warning: "N unreadable line(s) were skipped."
    Does not put removed text in any warning, log, or response.
    """
    kept_results = []
    removed_count = 0
    removed_raw_names: set[str] = set()

    for result in response.report.results:
        if result.test_name_canonical is not None:
            kept_results.append(result)
            continue

        raw = result.test_name_raw or ""
        raw_lower = raw.strip().lower()

        has_conf = result.confidence >= settings.OCR_VERIFY_BELOW
        has_3_letters = bool(re.search(r"[a-zA-Z]{3,}", raw))
        has_no_digits = not any(c.isdigit() for c in raw)
        has_no_forbidden_substrings = not any(sub in raw_lower for sub in FORBIDDEN_SUBSTRINGS)
        not_forbidden_exact = (
            raw_lower not in FORBIDDEN_EXACT
            and raw_lower.replace("-", " ") not in FORBIDDEN_EXACT
        )

        if (
            has_conf
            and has_3_letters
            and has_no_digits
            and has_no_forbidden_substrings
            and not_forbidden_exact
        ):
            kept_results.append(result)
        else:
            removed_count += 1
            if raw:
                removed_raw_names.add(raw)

    response.report.results = kept_results

    if removed_count > 0:
        cleaned_warnings = []
        for w in response.warnings:
            if not any(removed_name in w for removed_name in removed_raw_names):
                cleaned_warnings.append(w)
        response.warnings = cleaned_warnings
        response.warnings.append(f"{removed_count} unreadable line(s) were skipped.")

    return response


def _get_catalog_dict() -> dict[str, dict[str, Any]]:
    """Return dictionary of catalog entries keyed by canonical name."""
    catalog = load_test_catalog()
    return {entry["canonical_name"]: entry for entry in catalog}


def apply_ocr_safeguards(response: ExtractionResponse) -> ExtractionResponse:
    """Apply confidence thresholds and clinical plausibility guards to OCR results.

    Called AFTER process_report on OCR results only.
    Evaluated per result:
      - confidence < OCR_UNKNOWN_BELOW: flag = ResultFlag.UNKNOWN and warning added.
      - OCR_UNKNOWN_BELOW <= confidence < OCR_VERIFY_BELOW: flag kept, warning added.
      - plausibility guard for results with a canonical name: if converted value is outside
        [plausible_low, plausible_high], flag = ResultFlag.UNKNOWN and warning added.
      - Value is never modified.
      - Display name: canonical if available, otherwise raw.
    """
    catalog_dict = _get_catalog_dict()

    for result in response.report.results:
        display_name = result.test_name_canonical or result.test_name_raw

        # Unrecognized test name guard
        if result.test_name_canonical is None:
            result.flag = ResultFlag.UNKNOWN
            response.warnings.append(
                f'The test name "{result.test_name_raw}" was not recognized, '
                f'so this value was not compared with the reference range. '
                f'Please check it against your original report.'
            )
            continue

        # 1. Confidence checks
        if result.confidence < settings.OCR_UNKNOWN_BELOW:
            result.flag = ResultFlag.UNKNOWN
            response.warnings.append(
                f"Low OCR confidence for {display_name}. The value could not be read reliably, "
                f"so it has not been compared with the reference range. Please check it against your original report."
            )
        elif result.confidence < settings.OCR_VERIFY_BELOW:
            response.warnings.append(
                f"Please verify the value of {display_name} against your original report. The text may have been misread."
            )

        # 2. Plausibility guard for any OCR result with a canonical name
        if result.test_name_canonical and result.value is not None:
            entry = catalog_dict.get(result.test_name_canonical)
            if entry:
                p_low = entry.get("plausible_low")
                p_high = entry.get("plausible_high")
                if (p_low is not None and result.value < p_low) or (
                    p_high is not None and result.value > p_high
                ):
                    result.flag = ResultFlag.UNKNOWN
                    response.warnings.append(
                        f"The value of {display_name} looks unlikely and may have been misread. "
                        f"Please check it against your original report."
                    )

    return response
