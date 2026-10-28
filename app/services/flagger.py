"""Rule-based flagging service for laboratory test results."""

from functools import lru_cache
import re
from typing import Any

from app.data.catalog import load_test_catalog
from app.models.lab import LabResult, ResultFlag
from app.services.normalizer import canonicalize_name


@lru_cache(maxsize=1)
def _get_catalog_by_name() -> dict[str, dict[str, Any]]:
    """Index test catalog by canonical name."""
    catalog = load_test_catalog()
    return {entry["canonical_name"]: entry for entry in catalog}


def _is_non_numeric(text: str | None) -> bool:
    """Check if reference text is qualitative / non-numeric."""
    if not text or not isinstance(text, str):
        return False
    # If there are no digits, it's non-numeric
    return not bool(re.search(r"\d", text))


def compute_flag(
    result: LabResult, skip_critical: bool = False
) -> tuple[ResultFlag, str | None]:
    """Compute rule-based ResultFlag and optional warning for a LabResult.

    Rules:
    - Critical override: if beyond catalog critical_low/critical_high, return CRITICAL_LOW
      or CRITICAL_HIGH. Takes priority over LOW/HIGH, applies even without a reference range.
    - Numeric value with ref_low/ref_high:
        - below ref_low -> LOW
        - above ref_high -> HIGH
        - otherwise NORMAL
    - Strictness:
        - if ref_text starts with '<' (not '<='), value == ref_high is HIGH.
        - if ref_text starts with '>' (not '>='), value == ref_low is LOW.
        - for 'a - b' and '<=' / '>=' ranges, bounds are inclusive.
    - Non-numeric value_text:
        - if equals ref_text (case-insensitive) -> NORMAL.
        - if differs and ref_text is non-numeric -> UNKNOWN with warning.
    - No usable range and no critical rule -> UNKNOWN.
    """
    canon_name = result.test_name_canonical or canonicalize_name(result.test_name_raw)

    # 1. Critical override (if numeric and not skipped)
    if not skip_critical and result.value is not None and canon_name:
        catalog = _get_catalog_by_name()
        entry = catalog.get(canon_name)
        if entry:
            c_low = entry.get("critical_low")
            c_high = entry.get("critical_high")
            if c_low is not None and result.value < c_low:
                return ResultFlag.CRITICAL_LOW, None
            if c_high is not None and result.value > c_high:
                return ResultFlag.CRITICAL_HIGH, None

    # 2. Numeric range checks
    if result.value is not None:
        has_low = result.ref_low is not None
        has_high = result.ref_high is not None

        if not has_low and not has_high:
            return ResultFlag.UNKNOWN, None

        ref_t = (result.ref_text or "").strip()

        # Check HIGH condition
        if has_high:
            is_strict_lt = ref_t.startswith("<") and not ref_t.startswith("<=")
            if is_strict_lt:
                if result.value >= result.ref_high:
                    return ResultFlag.HIGH, None
            else:
                if result.value > result.ref_high:
                    return ResultFlag.HIGH, None

        # Check LOW condition
        if has_low:
            is_strict_gt = ref_t.startswith(">") and not ref_t.startswith(">=")
            if is_strict_gt:
                if result.value <= result.ref_low:
                    return ResultFlag.LOW, None
            else:
                if result.value < result.ref_low:
                    return ResultFlag.LOW, None

        return ResultFlag.NORMAL, None

    # 3. Non-numeric value_text check
    if result.value_text is not None:
        v_text = result.value_text.strip().lower()
        r_text = (result.ref_text or "").strip().lower()

        if not r_text:
            return ResultFlag.UNKNOWN, None

        if v_text == r_text:
            return ResultFlag.NORMAL, None

        if _is_non_numeric(result.ref_text):
            warning = "Non-numeric result differs from the reference. Please review with your doctor."
            return ResultFlag.UNKNOWN, warning

        return ResultFlag.UNKNOWN, None

    return ResultFlag.UNKNOWN, None
