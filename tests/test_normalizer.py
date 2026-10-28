"""Tests for test-name canonicalization."""

from app.data.catalog import load_test_catalog
from app.services.normalizer import canonicalize_name


def test_all_catalog_aliases_resolve_correctly():
    """Verify that every alias from test_catalog.json maps to its canonical name."""
    catalog = load_test_catalog()
    for entry in catalog:
        canonical_name = entry["canonical_name"]

        # Canonical name itself should resolve to canonical name
        assert canonicalize_name(canonical_name) == canonical_name

        # Each alias should resolve to canonical name
        for alias in entry.get("aliases", []):
            resolved = canonicalize_name(alias)
            assert resolved == canonical_name, f"Alias '{alias}' did not resolve to '{canonical_name}', got '{resolved}'"


def test_punctuation_case_and_spacing_variants():
    """Verify case-insensitivity, stripped punctuation, and flexible spacing."""
    assert canonicalize_name("LDL-C") == "LDL Cholesterol"
    assert canonicalize_name("ldl c") == "LDL Cholesterol"
    assert canonicalize_name("L.D.L") == "LDL Cholesterol"
    assert canonicalize_name("  ldl-c  ") == "LDL Cholesterol"
    assert canonicalize_name("Hb A1c") == "HbA1c"
    assert canonicalize_name("hba1c") == "HbA1c"
    assert canonicalize_name("HBA1C") == "HbA1c"
    assert canonicalize_name("Serum  Potassium") == "Potassium"
    assert canonicalize_name("k+") == "Potassium"
    assert canonicalize_name("K") == "Potassium"
    assert canonicalize_name("Protein, Urine") == "Urine Protein"
    assert canonicalize_name("urine protein") == "Urine Protein"


def test_unknown_name_returns_none():
    """Verify that unrecognized test names return None."""
    assert canonicalize_name("Random Nonexistent Test") is None
    assert canonicalize_name("") is None
    assert canonicalize_name(None) is None


def test_near_miss_returns_none_without_fuzzy_matching():
    """Verify that near-miss names like 'Hemoglobinn' return None (no fuzzy matching)."""
    assert canonicalize_name("Hemoglobinn") is None
    assert canonicalize_name("Glucoses") is None
    assert canonicalize_name("Cholesteroll") is None
    assert canonicalize_name("A1CC") is None
