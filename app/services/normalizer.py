"""Test-name normalization service."""

from functools import lru_cache
import re

from app.data.catalog import load_test_catalog


def clean_test_name(name: str) -> str:
    """Normalize a test name by lowercasing and removing punctuation, spaces, and hyphens."""
    if not name or not isinstance(name, str):
        return ""
    return re.sub(r"[^a-zA-Z0-9]", "", name).lower()


@lru_cache(maxsize=1)
def _get_name_mapping() -> dict[str, str]:
    """Build a mapping from cleaned name/alias strings to canonical test names."""
    catalog = load_test_catalog()
    mapping: dict[str, str] = {}

    for entry in catalog:
        canonical_name = entry["canonical_name"]
        canon_clean = clean_test_name(canonical_name)
        if canon_clean:
            mapping[canon_clean] = canonical_name

        for alias in entry.get("aliases", []):
            alias_clean = clean_test_name(alias)
            if alias_clean:
                mapping[alias_clean] = canonical_name

    return mapping


def canonicalize_name(raw: str | None) -> str | None:
    """Resolve a raw laboratory test name to its canonical name.

    Matching is case-insensitive and ignores punctuation, hyphens, and whitespace.
    Requires exact alias or canonical match only; no fuzzy matching is performed.
    Returns None for unknown names.
    """
    if not raw or not isinstance(raw, str):
        return None

    cleaned = clean_test_name(raw)
    if not cleaned:
        return None

    mapping = _get_name_mapping()
    return mapping.get(cleaned)
