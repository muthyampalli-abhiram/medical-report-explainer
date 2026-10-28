"""Processing pipeline connecting extraction, normalization, unit conversion, and flagging."""

import sys
from typing import Any

from app.data.catalog import load_test_catalog
from app.models.lab import ExtractionResponse
from app.services.flagger import compute_flag
from app.services.normalizer import canonicalize_name
from app.services.units import clean_unit, convert_to_canonical


def _get_catalog_dict() -> dict[str, dict[str, Any]]:
    """Return dictionary of catalog entries keyed by canonical name."""
    catalog = load_test_catalog()
    return {entry["canonical_name"]: entry for entry in catalog}


def process_report(response: ExtractionResponse) -> ExtractionResponse:
    """Process an extraction response by normalizing, converting units, and flagging.

    Order of operations for each result:
    (a) set test_name_canonical, adding warning "Unrecognized test name: <name>" if None
        and leaving it unflagged beyond the normal range check.
    (b) convert value, ref_low, and ref_high to canonical unit, storing value_original
        and unit_original. If the unit is not convertible, keep original units, skip the
        critical check, and add a warning.
    (c) compute the flag. Collect all warnings into the response.
    Never modifies patient data (there is none).
    """
    catalog_dict = _get_catalog_dict()

    for result in response.report.results:
        # Step (a): Test name canonicalization
        canonical_name = canonicalize_name(result.test_name_raw)
        result.test_name_canonical = canonical_name

        skip_critical = False
        if canonical_name is None:
            response.warnings.append(f"Unrecognized test name: {result.test_name_raw}")
            skip_critical = True

        # Step (b): Unit conversion
        if canonical_name and result.value is not None:
            entry = catalog_dict.get(canonical_name)
            canonical_unit = entry.get("canonical_unit") if entry else None

            if canonical_unit and result.unit is not None:
                # Check if conversion is needed
                if clean_unit(result.unit) != clean_unit(canonical_unit):
                    converted = convert_to_canonical(canonical_name, result.value, result.unit)
                    if converted is not None:
                        new_val, new_unit = converted
                        result.value_original = result.value
                        result.unit_original = result.unit
                        result.value = new_val
                        result.unit = new_unit

                        # Convert ref_low and ref_high using the original unit
                        if result.ref_low is not None and result.unit_original:
                            conv_low = convert_to_canonical(
                                canonical_name, result.ref_low, result.unit_original
                            )
                            if conv_low:
                                result.ref_low = conv_low[0]

                        if result.ref_high is not None and result.unit_original:
                            conv_high = convert_to_canonical(
                                canonical_name, result.ref_high, result.unit_original
                            )
                            if conv_high:
                                result.ref_high = conv_high[0]
                    else:
                        # Unit not convertible
                        response.warnings.append(
                            f"Unrecognized or inconvertible unit '{result.unit}' for test '{canonical_name}'."
                        )
                        skip_critical = True
                else:
                    # Normalize canonical unit casing
                    result.unit = canonical_unit

        # Step (c): Compute flag
        flag, flag_warning = compute_flag(result, skip_critical=skip_critical)
        result.flag = flag
        if flag_warning:
            response.warnings.append(flag_warning)

    return response


if __name__ == "__main__":
    from app.services.pdf_extractor import extract_report_from_pdf

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        extraction = extract_report_from_pdf(pdf_path)
        processed = process_report(extraction)

        print(f"{'Test':<25} {'Value':<12} {'Unit':<10} {'Reference':<18} {'Flag'}")
        print("-" * 75)
        for r in processed.report.results:
            name = r.test_name_canonical or r.test_name_raw
            val = str(r.value) if r.value is not None else (r.value_text or "")
            unit = r.unit or ""
            ref = r.ref_text or ""
            flag = r.flag.value
            print(f"{name:<25} {val:<12} {unit:<10} {ref:<18} {flag}")

        print("\nWarnings:")
        if processed.warnings:
            for w in processed.warnings:
                print(f"- {w}")
        else:
            print("- None")
    else:
        print("Usage: python -m app.services.pipeline <path_to_pdf>")
