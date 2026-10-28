"""Translations data loader and review status for medical report explanations."""

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

TRANSLATIONS_PATH = Path(__file__).resolve().parent / "translations.json"
SUPPORTED_LANGUAGES = {"en", "hi", "te"}


@lru_cache(maxsize=1)
def _load_translations_data() -> dict[str, Any]:
    """Load the translations.json file relative to the package location."""
    with open(TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_strings(lang: str) -> dict[str, str]:
    """Return dictionary of translation strings for 'en', 'hi', or 'te'.

    Raises ValueError for any unsupported language.
    """
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {lang}. Supported: en, hi, te")

    data = _load_translations_data()
    return data[lang]


def is_reviewed(lang: str) -> bool:
    """Return whether translations for given language have been verified/reviewed.

    English is always True.
    Other supported languages are read from _meta.
    Raises ValueError for any unsupported language.
    """
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {lang}. Supported: en, hi, te")

    if lang == "en":
        return True

    data = _load_translations_data()
    meta = data.get("_meta", {})
    return bool(meta.get(lang, {}).get("reviewed", False))
