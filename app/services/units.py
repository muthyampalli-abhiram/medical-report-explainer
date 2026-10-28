"""Unit conversion service for medical lab tests."""

from functools import lru_cache
from typing import Any

from app.data.catalog import load_test_catalog
from app.services.normalizer import canonicalize_name


def clean_unit(unit: str | None) -> str:
    """Normalize unit string for matching (case-insensitive, whitespace stripped)."""
    if not unit or not isinstance(unit, str):
        return ""
    return unit.strip().lower().replace(" ", "")


@lru_cache(maxsize=1)
def _get_catalog_by_name() -> dict[str, dict[str, Any]]:
    """Index test catalog by canonical name."""
    catalog = load_test_catalog()
    return {entry["canonical_name"]: entry for entry in catalog}


def convert_to_canonical(
    canonical_name: str, value: float | int, unit: str
) -> tuple[float, str] | None:
    """Convert a test value from a given unit to the test's canonical unit.

    Treats unit strings case-insensitively and ignores spaces.
    Rounds converted values to 2 decimal places.
    Returns None if the test or unit is unrecognized.
    """
    if value is None or unit is None:
        return None

    # Resolve name if needed
    canon = canonicalize_name(canonical_name) or canonical_name
    catalog = _get_catalog_by_name()
    entry = catalog.get(canon)

    if not entry:
        return None

    canonical_unit = entry.get("canonical_unit")
    if not canonical_unit:
        return None

    cleaned_input_unit = clean_unit(unit)
    cleaned_canon_unit = clean_unit(canonical_unit)

    # Already in canonical unit
    if cleaned_input_unit == cleaned_canon_unit:
        return round(float(value), 2), canonical_unit

    # Check conversions
    for conv in entry.get("unit_conversions", []):
        from_unit = conv.get("from_unit", "")
        if cleaned_input_unit == clean_unit(from_unit):
            # Special formula for HbA1c IFCC (mmol/mol) to DCCT (%):
            # percent = (mmol_per_mol / 10.929) + 2.15
            if canon == "HbA1c" and cleaned_input_unit == "mmol/mol":
                val_converted = (float(value) / 10.929) + 2.15
                return round(val_converted, 2), canonical_unit

            factor = conv.get("factor")
            if factor is not None:
                val_converted = float(value) * float(factor)
                return round(val_converted, 2), canonical_unit

    return None
