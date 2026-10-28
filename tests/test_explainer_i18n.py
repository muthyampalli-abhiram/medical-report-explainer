"""Unit tests for explanation i18n, number integrity, zero-LLM calls for Indic languages, and error handling."""

from pathlib import Path
import re
import pytest

from app.data.translations import get_strings
from app.models.lab import ExtractionResponse, LabReport, LabResult, ResultFlag
from app.services.explainer import explain_report
from app.services.llm_client import FakeLLMClient
from app.services.pdf_extractor import extract_report_from_pdf
from app.services.pipeline import process_report

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
REPORT_TABLE_PDF = SAMPLES_DIR / "report_table.pdf"
REPORT_LINES_PDF = SAMPLES_DIR / "report_lines.pdf"

NUMERIC_TOKEN_REGEX = re.compile(r"\b\d+(?:\.\d+)?\b")


@pytest.mark.parametrize("pdf_path", [REPORT_TABLE_PDF, REPORT_LINES_PDF])
def test_number_integrity_property(pdf_path: Path):
    """NUMBER-INTEGRITY PROPERTY: for report_table.pdf and report_lines.pdf, for every result,

    the list of numeric tokens (regex for integers and decimals) in the hi and te result_sentence
    equals the list in the English result_sentence, in the same order.
    """
    raw_extraction = extract_report_from_pdf(str(pdf_path))
    processed = process_report(raw_extraction)

    exp_en = explain_report(processed, FakeLLMClient(), language="en")
    exp_hi = explain_report(processed, FakeLLMClient(), language="hi")
    exp_te = explain_report(processed, FakeLLMClient(), language="te")

    assert len(exp_en.explanations) > 0
    assert len(exp_hi.explanations) == len(exp_en.explanations)
    assert len(exp_te.explanations) == len(exp_en.explanations)

    for i in range(len(exp_en.explanations)):
        en_sentence = exp_en.explanations[i].result_sentence
        hi_sentence = exp_hi.explanations[i].result_sentence
        te_sentence = exp_te.explanations[i].result_sentence

        en_tokens = NUMERIC_TOKEN_REGEX.findall(en_sentence)
        hi_tokens = NUMERIC_TOKEN_REGEX.findall(hi_sentence)
        te_tokens = NUMERIC_TOKEN_REGEX.findall(te_sentence)

        assert hi_tokens == en_tokens, (
            f"Result {i} ({exp_en.explanations[i].test_name}) numeric tokens mismatch between EN and HI:\n"
            f"EN: {en_sentence} -> {en_tokens}\nHI: {hi_sentence} -> {hi_tokens}"
        )
        assert te_tokens == en_tokens, (
            f"Result {i} ({exp_en.explanations[i].test_name}) numeric tokens mismatch between EN and TE:\n"
            f"EN: {en_sentence} -> {en_tokens}\nTE: {te_sentence} -> {te_tokens}"
        )


def test_flags_names_ordering_counts_and_unexplained_identical():
    """Flags, test names, ordering, abnormal counts, and unexplained_tests are identical across en, hi, te."""
    raw_extraction = extract_report_from_pdf(str(REPORT_TABLE_PDF))
    processed = process_report(raw_extraction)

    exp_en = explain_report(processed, FakeLLMClient(), language="en")
    exp_hi = explain_report(processed, FakeLLMClient(), language="hi")
    exp_te = explain_report(processed, FakeLLMClient(), language="te")

    en_names = [e.test_name for e in exp_en.explanations]
    hi_names = [e.test_name for e in exp_hi.explanations]
    te_names = [e.test_name for e in exp_te.explanations]
    assert hi_names == en_names
    assert te_names == en_names

    en_flags = [e.flag for e in exp_en.explanations]
    hi_flags = [e.flag for e in exp_hi.explanations]
    te_flags = [e.flag for e in exp_te.explanations]
    assert hi_flags == en_flags
    assert te_flags == en_flags

    assert exp_hi.unexplained_tests == exp_en.unexplained_tests
    assert exp_te.unexplained_tests == exp_en.unexplained_tests


def test_fake_llm_client_calls_empty_for_hi_and_te():
    """FakeLLMClient.calls is empty for hi and te, and non-empty for en when an LLM is provided."""
    raw_extraction = extract_report_from_pdf(str(REPORT_TABLE_PDF))
    processed = process_report(raw_extraction)

    llm_en = FakeLLMClient()
    explain_report(processed, llm_en, language="en")
    assert len(llm_en.calls) > 0, "Expected LLM calls for English explanation"

    llm_hi = FakeLLMClient()
    exp_hi = explain_report(processed, llm_hi, language="hi")
    assert len(llm_hi.calls) == 0, f"Expected 0 LLM calls for Hindi, got {len(llm_hi.calls)}"
    assert all(e.used_fallback for e in exp_hi.explanations)
    assert exp_hi.translation_reviewed is False
    assert exp_hi.draft_notice is not None

    llm_te = FakeLLMClient()
    exp_te = explain_report(processed, llm_te, language="te")
    assert len(llm_te.calls) == 0, f"Expected 0 LLM calls for Telugu, got {len(llm_te.calls)}"
    assert all(e.used_fallback for e in exp_te.explanations)
    assert exp_te.translation_reviewed is False
    assert exp_te.draft_notice is not None


def test_critical_potassium_case_escalation_and_overall_notice():
    """A critical potassium 7.0 case gets the translated escalation and critical notice in hi and te, and English in en."""
    report = LabReport(
        results=[
            LabResult(
                test_name_raw="Potassium",
                test_name_canonical="Potassium",
                value=7.0,
                unit="mmol/L",
                ref_low=3.5,
                ref_high=5.0,
                ref_text="3.5 - 5.0",
                flag=ResultFlag.CRITICAL_HIGH,
            )
        ]
    )
    response = ExtractionResponse(report=report)

    en_strings = get_strings("en")
    hi_strings = get_strings("hi")
    te_strings = get_strings("te")

    exp_en = explain_report(response, FakeLLMClient(), language="en")
    assert exp_en.explanations[0].escalation == en_strings["escalation"]
    assert exp_en.overall_notice == en_strings["overall_notice_critical"]

    exp_hi = explain_report(response, FakeLLMClient(), language="hi")
    assert exp_hi.explanations[0].escalation == hi_strings["escalation"]
    assert exp_hi.overall_notice == hi_strings["overall_notice_critical"]

    exp_te = explain_report(response, FakeLLMClient(), language="te")
    assert exp_te.explanations[0].escalation == te_strings["escalation"]
    assert exp_te.overall_notice == te_strings["overall_notice_critical"]


def test_disclaimer_present_and_translated_in_every_output():
    """The disclaimer is present and translated in every output."""
    raw_extraction = extract_report_from_pdf(str(REPORT_LINES_PDF))
    processed = process_report(raw_extraction)

    en_strings = get_strings("en")
    hi_strings = get_strings("hi")
    te_strings = get_strings("te")

    exp_en = explain_report(processed, FakeLLMClient(), language="en")
    assert exp_en.disclaimer == en_strings["disclaimer"]

    exp_hi = explain_report(processed, FakeLLMClient(), language="hi")
    assert exp_hi.disclaimer == hi_strings["disclaimer"]

    exp_te = explain_report(processed, FakeLLMClient(), language="te")
    assert exp_te.disclaimer == te_strings["disclaimer"]


def test_unsupported_language_raises_value_error():
    """Requesting an unsupported language in the service raises ValueError."""
    raw_extraction = extract_report_from_pdf(str(REPORT_TABLE_PDF))
    processed = process_report(raw_extraction)

    for invalid_lang in ("xx", "fr", "es", "de", "unknown", ""):
        with pytest.raises(ValueError, match="Unsupported language"):
            explain_report(processed, FakeLLMClient(), language=invalid_lang)
