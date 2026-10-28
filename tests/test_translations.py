"""Unit tests for translations data integrity, placeholders, scripts, and review metadata."""

import json
from pathlib import Path
import re
import pytest

from app.data.catalog import load_test_catalog
from app.data.translations import get_strings, is_reviewed

TRANSLATIONS_PATH = Path(__file__).resolve().parent.parent / "app" / "data" / "translations.json"


def test_translations_json_keys_and_placeholders_match():
    """hi and te contain exactly the same keys as en, and each string has identical {placeholders}."""
    with open(TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "en" in data
    assert "hi" in data
    assert "te" in data
    assert "_meta" in data

    en = data["en"]
    hi = data["hi"]
    te = data["te"]

    required_keys = {
        "result_above",
        "result_below",
        "result_within",
        "result_text",
        "result_unknown",
        "escalation",
        "overall_notice",
        "overall_notice_critical",
        "disclaimer",
        "draft_notice",
    }

    assert set(en.keys()) == required_keys
    assert set(hi.keys()) == required_keys
    assert set(te.keys()) == required_keys

    placeholder_pattern = re.compile(r"\{([a-zA-Z_]+)\}")

    for key in required_keys:
        en_placeholders = set(placeholder_pattern.findall(en[key]))
        hi_placeholders = set(placeholder_pattern.findall(hi[key]))
        te_placeholders = set(placeholder_pattern.findall(te[key]))

        assert hi_placeholders == en_placeholders, f"Placeholder mismatch in 'hi' for key: {key}"
        assert te_placeholders == en_placeholders, f"Placeholder mismatch in 'te' for key: {key}"


def test_no_digits_in_hi_or_te_translations():
    """No ASCII digits (0-9) and no Devanagari or Telugu digits appear in any hi/te translation string."""
    with open(TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    digit_regex = re.compile(r"[0-9\u0966-\u096F\u0C66-\u0C6F]")

    for lang in ("hi", "te"):
        for key, text in data[lang].items():
            match = digit_regex.search(text)
            assert match is None, f"Found digit '{match.group()}' in {lang}[{key}]: {text}"


def test_script_characters_present_and_not_equal_to_english():
    """hi strings contain Devanagari characters, te strings contain Telugu characters, and none equal English."""
    with open(TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    en = data["en"]
    hi = data["hi"]
    te = data["te"]

    devanagari_regex = re.compile(r"[\u0900-\u097F]")
    telugu_regex = re.compile(r"[\u0C00-\u0C7F]")

    for key in en:
        hi_text = hi[key]
        te_text = te[key]
        en_text = en[key]

        assert devanagari_regex.search(hi_text) is not None, f"No Devanagari characters in hi[{key}]"
        assert telugu_regex.search(te_text) is not None, f"No Telugu characters in te[{key}]"

        assert hi_text != en_text, f"hi[{key}] equals en[{key}]"
        assert te_text != en_text, f"te[{key}] equals en[{key}]"


def test_canonical_test_descriptions_in_catalog():
    """Every canonical test in catalog has non-empty plain_description_hi and plain_description_te with script chars."""
    catalog = load_test_catalog()
    assert len(catalog) == 9

    devanagari_regex = re.compile(r"[\u0900-\u097F]")
    telugu_regex = re.compile(r"[\u0C00-\u0C7F]")

    for entry in catalog:
        name = entry["canonical_name"]
        hi_desc = entry.get("plain_description_hi")
        te_desc = entry.get("plain_description_te")

        assert isinstance(hi_desc, str) and len(hi_desc.strip()) > 0, f"Missing plain_description_hi for {name}"
        assert isinstance(te_desc, str) and len(te_desc.strip()) > 0, f"Missing plain_description_te for {name}"

        assert devanagari_regex.search(hi_desc) is not None, f"No Devanagari characters in hi desc for {name}"
        assert telugu_regex.search(te_desc) is not None, f"No Telugu characters in te desc for {name}"

        # Test name should appear in Latin letters inside the translation
        assert name.lower() in hi_desc.lower(), f"Canonical test name {name} not found in plain_description_hi"
        assert name.lower() in te_desc.lower(), f"Canonical test name {name} not found in plain_description_te"


def test_meta_reviewed_status_and_helper_functions():
    """_meta shows reviewed false for hi and te, is_reviewed helper returns correct bool, raises ValueError on bad lang."""
    with open(TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("_meta", {})
    assert meta.get("hi", {}).get("reviewed") is False
    assert meta.get("te", {}).get("reviewed") is False

    assert is_reviewed("en") is True
    assert is_reviewed("hi") is False
    assert is_reviewed("te") is False

    with pytest.raises(ValueError, match="Unsupported language"):
        is_reviewed("fr")

    with pytest.raises(ValueError, match="Unsupported language"):
        get_strings("es")


def test_english_strings_match_pre_refactor_baseline():
    """English strings in translations.json match known explainer sentences and baseline text."""
    en_strings = get_strings("en")

    # HbA1c sentence
    formatted_hba1c = en_strings["result_above"].format(
        value="7.4", unit="%", range="4.0 - 5.6 %"
    )
    assert (
        formatted_hba1c
        == "Your result is 7.4 %. The lab's reference range is 4.0 - 5.6 %. This is above the reference range."
    )

    # Disclaimer
    assert (
        en_strings["disclaimer"]
        == "This tool explains lab reports in general terms and does not provide a diagnosis or medical advice. Always discuss your results with your doctor."
    )

    # Escalation
    assert (
        en_strings["escalation"]
        == "This value is far outside the usual range. Please contact your doctor or a healthcare provider promptly."
    )

    # Critical notice
    assert (
        en_strings["overall_notice_critical"]
        == "One or more test results are far outside the usual range and need prompt medical attention. Please contact your doctor or a healthcare provider promptly."
    )

    # Draft notices are non-empty strings
    for lang in ("en", "hi", "te"):
        strings = get_strings(lang)
        assert "draft_notice" in strings
        assert len(strings["draft_notice"].strip()) > 0


def test_wording_fixes_urgency_and_telugu_terms():
    """Verify urgency matching (no 'तुरंत' or 'వెంటనే'), no 'ప్రమాణం' in te, and 'రిఫరెన్స్' in te result strings."""
    hi_strings = get_strings("hi")
    te_strings = get_strings("te")

    # hi escalation and overall_notice_critical do not contain "तुरंत"
    assert "तुरंत" not in hi_strings["escalation"]
    assert "तुरंत" not in hi_strings["overall_notice_critical"]

    # te escalation and overall_notice_critical do not contain "వెంటనే"
    assert "వెంటనే" not in te_strings["escalation"]
    assert "వెంటనే" not in te_strings["overall_notice_critical"]

    # no te string contains "ప్రమాణం"
    for key, text in te_strings.items():
        assert "ప్రమాణం" not in text, f"Found 'ప్రమాణం' in te[{key}]: {text}"

    # te result_above/below/within strings contain "రిఫరెన్స్"
    assert "రిఫరెన్స్" in te_strings["result_above"]
    assert "రిఫరెన్స్" in te_strings["result_below"]
    assert "రిఫరెన్స్" in te_strings["result_within"]

