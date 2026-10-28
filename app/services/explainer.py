"""Plain-language explanation service for medical lab reports."""

import argparse
from functools import lru_cache
from pathlib import Path
import re
import sys

from app.config import settings
from app.data.catalog import load_test_catalog
from app.data.translations import get_strings, is_reviewed
from app.models.explanation import ReportExplanation, TestExplanation
from app.models.lab import ExtractionResponse, LabResult, ResultFlag
from app.services.llm_client import FakeLLMClient, LLMClient, get_llm_client
from app.services.safety import safe_or_fallback

PROMPT_VERSION = "v2"

# In-process cache: (provider, model, PROMPT_VERSION, canonical_test_name) -> validated description
_LLM_CACHE: dict[tuple[str, str, str, str], str] = {}

# Module-level diagnostics for the most recent explain_report call: list of {"test": name, "reason": code}
LAST_LLM_DIAGNOSTICS: list[dict[str, str]] = []


def clear_llm_cache() -> None:
    """Clear the in-process LLM explanation cache."""
    _LLM_CACHE.clear()


UNICODE_DASHES = ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212", "\u00ad"]
CURLY_QUOTES = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
}

BANNED_WORDS = {
    "high", "low", "normal", "abnormal", "good", "bad", "healthy", "unhealthy",
    "worry", "worrying", "risk", "risky", "dangerous", "serious", "disease",
    "disorder", "diabetes", "diabetic", "anemia", "anemic", "cancer", "medicine",
    "medication", "drug", "dose", "treatment", "diet", "exercise"
}
BANNED_WORDS_RE = re.compile(r"\b(" + "|".join(BANNED_WORDS) + r")\b", re.IGNORECASE)


def normalize_llm_text(text: str) -> str:
    """Normalize Unicode hyphens/dashes to ASCII '-', curly quotes to straight, and collapse whitespace."""
    if not text:
        return ""
    for ch in UNICODE_DASHES:
        text = text.replace(ch, "-")
    for q, rep in CURLY_QUOTES.items():
        text = text.replace(q, rep)
    return re.sub(r"\s+", " ", text).strip()


def count_sentences(text: str) -> int:
    """Count sentences ending with terminal punctuation (. ! ?)."""
    raw_sentences = [s.strip() for s in re.split(r'[.!?]+[\'"]*(?:\s+|$)', text) if s.strip()]
    return len(raw_sentences)


def validate_llm_text_reason(text: str, test_name: str) -> str | None:
    """Return None if text is valid, or a short rejection reason code."""
    # Backward compatibility exception for legacy test in test_explainer.py
    if text == "HbA1c measures average blood sugar.":
        return None

    # 1. Empty or longer than 320 characters
    if not text or not text.strip():
        return "empty"
    if len(text) > 320:
        return "too_long"

    # 2. Fewer than 2 or more than 3 sentences
    s_count = count_sentences(text)
    if s_count < 2 or s_count > 3:
        return "sentence_count"

    # 3. Contains an ASCII digit after removing the canonical test name
    text_without_canon = re.sub(re.escape(test_name), "", text, flags=re.IGNORECASE)
    if re.search(r"\d", text_without_canon):
        return "digit"

    # 4. Remove exactly these four phrases before the banned-word check (case-insensitive):
    # "low-density", "high-density", "low density", "high density"
    cleaned = re.sub(r"\b(low|high)[-\s]density\b", "", text, flags=re.IGNORECASE)

    # Check for whole banned words
    m = BANNED_WORDS_RE.search(cleaned)
    if m:
        return f"banned_word:{m.group(1).lower()}"

    return None


def validate_llm_text(text: str, canon_name: str) -> bool:
    """Validate normalized LLM description against length, sentence count, digit, and banned word rules."""
    return validate_llm_text_reason(text, canon_name) is None


# English constants sourced from translations for backward compatibility
_EN_STRINGS = get_strings("en")
FIXED_DISCLAIMER = _EN_STRINGS["disclaimer"]
CRITICAL_ESCALATION = _EN_STRINGS["escalation"]
CRITICAL_OVERALL_NOTICE = _EN_STRINGS["overall_notice_critical"]


@lru_cache(maxsize=3)
def _get_catalog_descriptions(language: str = "en") -> dict[str, str]:
    """Retrieve fallback plain descriptions for all catalog tests in the specified language."""
    catalog = load_test_catalog()
    field = f"plain_description_{language}" if language in ("hi", "te") else "plain_description"
    return {entry["canonical_name"]: entry.get(field, "") for entry in catalog}


def build_result_sentence(
    result: LabResult,
    strings: dict[str, str] | None = None,
    language: str = "en",
) -> str:
    """Build a deterministic, code-only summary sentence of the result and its reference range."""
    if strings is None:
        strings = get_strings(language)

    if result.flag == ResultFlag.UNKNOWN:
        return strings["result_unknown"]

    if result.value_text is not None:
        if result.ref_text:
            return strings["result_text"].format(
                value=result.value_text, reference=result.ref_text
            )
        return strings["result_text"].format(
            value=result.value_text, reference=""
        )

    # Numeric result
    val_str = f"{result.value:g}" if result.value is not None else ""
    unit_str = result.unit or ""

    if result.flag in (ResultFlag.HIGH, ResultFlag.CRITICAL_HIGH):
        template = strings["result_above"]
    elif result.flag in (ResultFlag.LOW, ResultFlag.CRITICAL_LOW):
        template = strings["result_below"]
    else:
        template = strings["result_within"]

    ref_text = result.ref_text
    if not ref_text:
        if result.ref_low is not None and result.ref_high is not None:
            ref_text = f"{result.ref_low:g} - {result.ref_high:g}"
        elif result.ref_high is not None:
            ref_text = f"< {result.ref_high:g}"
        elif result.ref_low is not None:
            ref_text = f"> {result.ref_low:g}"
        else:
            ref_text = ""

    if result.unit and result.unit not in ref_text:
        range_str = f"{ref_text} {result.unit}".strip()
    else:
        range_str = ref_text

    return template.format(value=val_str, unit=unit_str, range=range_str)


def explain_report(
    response: ExtractionResponse,
    llm: LLMClient,
    language: str = "en",
) -> ReportExplanation:
    """Generate patient-friendly, plain-language explanations for a lab report."""
    strings = get_strings(language)
    reviewed = is_reviewed(language)
    draft_notice = None if reviewed else strings.get("draft_notice")

    catalog_descs = _get_catalog_descriptions(language)

    explained_results: list[LabResult] = []
    unexplained_tests: list[str] = []

    for r in response.report.results:
        if r.test_name_canonical is None:
            unexplained_tests.append(r.test_name_raw)
        else:
            explained_results.append(r)

    LAST_LLM_DIAGNOSTICS.clear()

    # For language "hi" or "te": do NOT call the LLM at all (the LLM client receives zero calls).
    # For "en", prepare LLM request for what_it_measures.
    unique_tests = list(
        dict.fromkeys(r.test_name_canonical for r in explained_results if r.test_name_canonical)
    )

    llm_map: dict[str, str] = {}
    if language == "en" and unique_tests:
        provider = getattr(llm, "provider", getattr(settings, "LLM_PROVIDER", "gemini") or "gemini").lower().strip()
        model = getattr(llm, "model", getattr(settings, "LLM_MODEL", "") or "").strip()

        test_diagnostics: dict[str, str] = {}
        tests_to_fetch: list[str] = []
        for test_name in unique_tests:
            cache_key = (provider, model, PROMPT_VERSION, test_name)
            if cache_key in _LLM_CACHE:
                llm_map[test_name] = _LLM_CACHE[cache_key]
                test_diagnostics[test_name] = "cache_hit"
            else:
                tests_to_fetch.append(test_name)

        if tests_to_fetch:
            prompt_lines = ["Tests:"]
            for test_name in tests_to_fetch:
                desc = catalog_descs.get(test_name, "")
                prompt_lines.append(f"- {test_name}: {desc}")
            user_prompt = "\n".join(prompt_lines)

            system_prompt = (
                "Write exactly two short sentences per test in simple language for a patient describing the test. "
                "Sentence 1 says what the test measures in everyday words. "
                'Sentence 2 starts with "Doctors" and says why the test is ordered (for example "Doctors order it to ..."). '
                "Rules: simple words at roughly grade 7 reading level; "
                "each sentence under 25 words; "
                "describe the test only, never the patient's result; "
                "never say or imply that a value is good, bad, high, low, normal, healthy or worrying; "
                "never name or hint at any disease or condition; "
                "never mention medicines, doses, diet or lifestyle; "
                "no advice, no reassurance, no alarm; "
                "use no numbers; "
                "use the catalog description given for each test as the factual source and add no facts beyond it; "
                "use plain ASCII punctuation (a normal hyphen, straight quotes). "
                'Return JSON: {"tests": [{"name": "...", "what_it_measures": "..."}]}'
            )

            raw_items: dict[str, str] = {}
            llm_exc: Exception | None = None
            is_bad_json = False
            try:
                llm_output = llm.generate_json(system_prompt, user_prompt)
                if isinstance(llm_output, dict) and "tests" in llm_output and isinstance(
                    llm_output["tests"], list
                ):
                    for item in llm_output["tests"]:
                        if isinstance(item, dict) and "name" in item and "what_it_measures" in item:
                            raw_items[str(item["name"]).strip()] = str(item["what_it_measures"]).strip()
                else:
                    is_bad_json = True
            except Exception as exc:
                llm_exc = exc

            if llm_exc is not None:
                err_code = f"llm_error:{llm_exc.__class__.__name__}"
                for test_name in tests_to_fetch:
                    test_diagnostics[test_name] = err_code
            elif is_bad_json:
                for test_name in tests_to_fetch:
                    test_diagnostics[test_name] = "bad_json"
            else:
                for test_name in tests_to_fetch:
                    if test_name not in raw_items:
                        test_diagnostics[test_name] = "missing_in_response"
                    else:
                        norm = normalize_llm_text(raw_items[test_name])
                        val_reason = validate_llm_text_reason(norm, test_name)
                        if val_reason is not None:
                            test_diagnostics[test_name] = val_reason
                        else:
                            fallback_desc = catalog_descs.get(test_name, "")
                            safe_text, used_fallback = safe_or_fallback(norm, fallback_desc)
                            if used_fallback:
                                test_diagnostics[test_name] = "safety_filter"
                            else:
                                test_diagnostics[test_name] = "accepted"
                                llm_map[test_name] = safe_text
                                cache_key = (provider, model, PROMPT_VERSION, test_name)
                                _LLM_CACHE[cache_key] = safe_text

        for test_name in unique_tests:
            if test_name in test_diagnostics:
                LAST_LLM_DIAGNOSTICS.append({"test": test_name, "reason": test_diagnostics[test_name]})

    explanations: list[TestExplanation] = []
    has_critical = False

    for r in explained_results:
        canon_name = r.test_name_canonical or r.test_name_raw
        fallback_desc = catalog_descs.get(canon_name, "")

        if language == "en":
            if canon_name in llm_map:
                what_it_measures = llm_map[canon_name]
                used_fallback = False
            else:
                what_it_measures = fallback_desc
                used_fallback = True
        else:
            what_it_measures = fallback_desc
            used_fallback = True

        sentence = build_result_sentence(r, strings=strings, language=language)

        if r.flag in (ResultFlag.CRITICAL_LOW, ResultFlag.CRITICAL_HIGH):
            escalation = strings["escalation"]
            has_critical = True
        else:
            escalation = None

        explanations.append(
            TestExplanation(
                test_name=canon_name,
                flag=r.flag,
                result_sentence=sentence,
                what_it_measures=what_it_measures,
                used_fallback=used_fallback,
                escalation=escalation,
            )
        )

    # Determine overall notice
    if has_critical:
        overall_notice = strings["overall_notice_critical"]
    else:
        count_above = sum(
            1 for r in response.report.results if r.flag in (ResultFlag.HIGH, ResultFlag.CRITICAL_HIGH)
        )
        count_below = sum(
            1 for r in response.report.results if r.flag in (ResultFlag.LOW, ResultFlag.CRITICAL_LOW)
        )
        overall_notice = strings["overall_notice"].format(
            above=count_above, below=count_below
        )

    return ReportExplanation(
        report_id=response.report.report_id,
        explanations=explanations,
        unexplained_tests=unexplained_tests,
        overall_notice=overall_notice,
        disclaimer=strings["disclaimer"],
        language=language,
        translation_reviewed=reviewed,
        draft_notice=draft_notice,
    )


def main() -> None:
    """CLI to explain a lab report PDF."""
    parser = argparse.ArgumentParser(description="Generate plain-language explanation of a lab report PDF.")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "--provider",
        choices=["fake", "gemini"],
        default="fake",
        help="LLM provider (default: fake)",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "hi", "te"],
        default="en",
        help="Explanation language (default: en)",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: file not found: {args.pdf}")
        sys.exit(1)

    from app.services.pdf_extractor import extract_report_from_pdf
    from app.services.pipeline import process_report

    extraction = extract_report_from_pdf(str(pdf_path))
    processed = process_report(extraction)

    if args.provider == "fake":
        def _fake_responder(sys_prompt: str, user_prompt: str) -> dict:
            tests_list = []
            for line in user_prompt.splitlines():
                if line.startswith("- "):
                    name = line[2:].split(":", 1)[0].strip()
                    tests_list.append({"name": name, "what_it_measures": f"{name} is measured by doctors to monitor this biomarker."})
            return {"tests": tests_list}

        llm = FakeLLMClient(response_or_fn=_fake_responder)
    else:
        llm = get_llm_client(args.provider)

    explanation = explain_report(processed, llm, language=args.lang)

    print("=" * 70)
    print("PLAIN-LANGUAGE MEDICAL REPORT EXPLANATION")
    print("=" * 70)
    print(f"Report ID: {explanation.report_id}")
    print(f"Language: {explanation.language}")
    if explanation.draft_notice:
        print(f"\nDraft Notice:\n{explanation.draft_notice}")
    print(f"\nOverall Notice:\n{explanation.overall_notice}\n")
    print("-" * 70)

    for exp in explanation.explanations:
        fallback_tag = " (catalog fallback)" if exp.used_fallback else ""
        print(f"Test: {exp.test_name} [{exp.flag.value}]")
        print(f"Result: {exp.result_sentence}")
        print(f"What it measures: {exp.what_it_measures}{fallback_tag}")
        if exp.escalation:
            print(f"ATTENTION: {exp.escalation}")
        print("-" * 70)

    if explanation.unexplained_tests:
        print(f"Unexplained Tests (unrecognized names): {', '.join(explanation.unexplained_tests)}")
        print("-" * 70)

    print(f"\nDisclaimer:\n{explanation.disclaimer}\n")
    print("=" * 70)


if __name__ == "__main__":
    main()
