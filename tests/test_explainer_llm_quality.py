"""Tests for LLM explanation quality, validation, normalization, and caching."""

from pathlib import Path
import pytest

from app.services.explainer import clear_llm_cache, explain_report
from app.services.llm_client import FakeLLMClient
from app.services.pdf_extractor import extract_report_from_pdf
from app.services.pipeline import process_report

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture(autouse=True)
def _reset_cache():
    clear_llm_cache()
    yield
    clear_llm_cache()


def test_good_two_sentence_outputs_accepted():
    """Good two-sentence outputs starting sentence 2 with 'Doctors' are accepted with used_fallback=False."""
    raw_extraction = extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf"))
    processed = process_report(raw_extraction)

    good_text = (
        "HbA1c measures average blood sugar levels over recent months. "
        "Doctors order it to check blood sugar patterns over time."
    )
    fake_llm = FakeLLMClient(
        response_or_fn={
            "tests": [
                {"name": "HbA1c", "what_it_measures": good_text}
            ]
        }
    )

    explanation = explain_report(processed, fake_llm)
    hba1c_exp = next(e for e in explanation.explanations if e.test_name == "HbA1c")
    assert hba1c_exp.used_fallback is False
    assert hba1c_exp.what_it_measures == good_text


def test_ldl_output_ascii_and_unicode_hyphen_accepted():
    """Verify LDL output is accepted with ASCII and Unicode non-breaking hyphen (U+2011) stored as ASCII."""
    raw_extraction = extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf"))
    processed = process_report(raw_extraction)

    ascii_text = (
        "LDL cholesterol measures the cholesterol in low-density lipoprotein particles. "
        "Doctors order it to look at blood fats."
    )
    fake_llm_ascii = FakeLLMClient(
        response_or_fn={"tests": [{"name": "LDL Cholesterol", "what_it_measures": ascii_text}]}
    )
    exp_ascii = explain_report(processed, fake_llm_ascii)
    ldl_ascii = next(e for e in exp_ascii.explanations if e.test_name == "LDL Cholesterol")
    assert ldl_ascii.used_fallback is False
    assert ldl_ascii.what_it_measures == ascii_text

    clear_llm_cache()

    unicode_text = (
        "LDL cholesterol measures the cholesterol in low\u2011density lipoprotein particles. "
        "Doctors order it to look at blood fats."
    )
    fake_llm_unicode = FakeLLMClient(
        response_or_fn={"tests": [{"name": "LDL Cholesterol", "what_it_measures": unicode_text}]}
    )
    exp_unicode = explain_report(processed, fake_llm_unicode)
    ldl_unicode = next(e for e in exp_unicode.explanations if e.test_name == "LDL Cholesterol")
    assert ldl_unicode.used_fallback is False
    assert "\u2011" not in ldl_unicode.what_it_measures
    assert ldl_unicode.what_it_measures == ascii_text


@pytest.mark.parametrize(
    "bad_text,test_name",
    [
        # One sentence only
        ("Fasting glucose measures blood sugar levels after an overnight fast.", "Fasting Glucose"),
        # Four sentences
        ("One sentence. Two sentences. Three sentences. Four sentences.", "Fasting Glucose"),
        # 400 characters
        ("A" * 350 + ". Doctors order it to check your health levels.", "Fasting Glucose"),
        # Digit ("over 3 months" rejected)
        ("HbA1c measures blood sugar over 3 months. Doctors order it to monitor sugar trends.", "HbA1c"),
        # Banned word "high" on its own
        ("This test checks for high levels of lipids in your blood. Doctors order it for screening.", "Total Cholesterol"),
        # Banned word "healthy"
        ("This test measures healthy particles circulating in blood. Doctors order it to check status.", "HDL Cholesterol"),
        # Empty text
        ("", "Fasting Glucose"),
    ],
)
def test_rejection_rules_fallback_to_catalog(bad_text, test_name):
    """Each rejection rule falls back to the catalog description with used_fallback=True."""
    raw_extraction = extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf"))
    processed = process_report(raw_extraction)

    fake_llm = FakeLLMClient(
        response_or_fn={"tests": [{"name": test_name, "what_it_measures": bad_text}]}
    )
    explanation = explain_report(processed, fake_llm)
    exp = next(e for e in explanation.explanations if e.test_name == test_name)
    assert exp.used_fallback is True
    assert exp.what_it_measures != bad_text


def test_digit_inside_hba1c_name_allowed():
    """Verify digits inside canonical name 'HbA1c' are allowed when no other digits are present."""
    raw_extraction = extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf"))
    processed = process_report(raw_extraction)

    good_text = (
        "HbA1c measures average blood sugar levels over recent months. "
        "Doctors order it to evaluate sugar patterns over time."
    )
    fake_llm = FakeLLMClient(
        response_or_fn={"tests": [{"name": "HbA1c", "what_it_measures": good_text}]}
    )
    explanation = explain_report(processed, fake_llm)
    hba1c_exp = next(e for e in explanation.explanations if e.test_name == "HbA1c")
    assert hba1c_exp.used_fallback is False
    assert hba1c_exp.what_it_measures == good_text


def test_user_prompt_privacy_no_leaked_metadata():
    """Verify user prompt contains canonical names and catalog descriptions, and none of the sensitive tokens."""
    raw_extraction = extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf"))
    processed = process_report(raw_extraction)

    fake_llm = FakeLLMClient(response_or_fn={"tests": []})
    explain_report(processed, fake_llm)

    assert len(fake_llm.calls) == 1
    user_prompt = fake_llm.calls[0]["user_prompt"]

    assert "HbA1c" in user_prompt
    assert "Fasting Glucose" in user_prompt
    assert "average blood sugar" in user_prompt

    forbidden = ["7.4", "132", "mg/dL", "Sample Diagnostics Lab", "report_table"]
    for token in forbidden:
        assert token not in user_prompt, f"Forbidden token '{token}' found in user prompt"


def test_cache_second_call_zero_llm_calls_and_identical_wording():
    """A second explain_report call for the same tests makes zero LLM calls and returns identical wording."""
    raw_extraction = extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf"))
    processed = process_report(raw_extraction)

    all_tests_payload = [
        {
            "name": r.test_name_canonical,
            "what_it_measures": f"{r.test_name_canonical} measures specific indicators in the blood. Doctors order it to monitor overall physical status.",
        }
        for r in processed.report.results
        if r.test_name_canonical
    ]

    fake_llm = FakeLLMClient(response_or_fn={"tests": all_tests_payload})

    exp1 = explain_report(processed, fake_llm)
    assert len(fake_llm.calls) == 1

    # Second call with the same processed report
    exp2 = explain_report(processed, fake_llm)
    assert len(fake_llm.calls) == 1  # FakeLLMClient.calls does not grow!

    for e1, e2 in zip(exp1.explanations, exp2.explanations):
        assert e1.what_it_measures == e2.what_it_measures
        assert e1.used_fallback is False and e2.used_fallback is False


def test_cache_fallback_not_cached_and_retried():
    """A fallback is not cached, so the next call asks the LLM again."""
    raw_extraction = extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf"))
    processed = process_report(raw_extraction)

    # First call returns banned word for Fasting Glucose (fallback)
    bad_llm = FakeLLMClient(
        response_or_fn={
            "tests": [
                {"name": "Fasting Glucose", "what_it_measures": "Fasting glucose checks for diabetes. Doctors order it."}
            ]
        }
    )
    exp1 = explain_report(processed, bad_llm)
    g1 = next(e for e in exp1.explanations if e.test_name == "Fasting Glucose")
    assert g1.used_fallback is True
    assert len(bad_llm.calls) == 1

    # Second call with valid text asks the LLM again because fallback was not cached
    good_text = (
        "Fasting glucose measures sugar in the blood after an overnight fast. "
        "Doctors order it to check how the body handles sugar."
    )
    good_llm = FakeLLMClient(
        response_or_fn={"tests": [{"name": "Fasting Glucose", "what_it_measures": good_text}]}
    )
    exp2 = explain_report(processed, good_llm)
    assert len(good_llm.calls) == 1
    g2 = next(e for e in exp2.explanations if e.test_name == "Fasting Glucose")
    assert g2.used_fallback is False
    assert g2.what_it_measures == good_text


def test_cache_partial_asks_only_for_missing():
    """A partial cache asks the LLM only for the missing tests."""
    raw_extraction = extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf"))
    processed = process_report(raw_extraction)

    text_hba1c = (
        "HbA1c measures average blood sugar levels over recent months. "
        "Doctors order it to check blood sugar patterns over time."
    )
    # Prime cache with HbA1c only
    llm1 = FakeLLMClient(
        response_or_fn={"tests": [{"name": "HbA1c", "what_it_measures": text_hba1c}]}
    )
    explain_report(processed, llm1)

    # Now make second call with a new client
    text_glucose = (
        "Fasting glucose measures sugar in the blood after an overnight fast. "
        "Doctors order it to check how the body handles sugar."
    )
    llm2 = FakeLLMClient(
        response_or_fn={"tests": [{"name": "Fasting Glucose", "what_it_measures": text_glucose}]}
    )
    explain_report(processed, llm2)

    assert len(llm2.calls) == 1
    prompt2 = llm2.calls[0]["user_prompt"]
    # HbA1c is already cached, so it must NOT be in the user prompt!
    assert "HbA1c:" not in prompt2
    # Fasting Glucose was not cached, so it must be requested
    assert "Fasting Glucose:" in prompt2


def test_clear_llm_cache_empties_it():
    """clear_llm_cache() empties the cache so subsequent calls re-query LLM."""
    raw_extraction = extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf"))
    processed = process_report(raw_extraction)

    text_hba1c = (
        "HbA1c measures average blood sugar levels over recent months. "
        "Doctors order it to check blood sugar patterns over time."
    )
    llm1 = FakeLLMClient(
        response_or_fn={"tests": [{"name": "HbA1c", "what_it_measures": text_hba1c}]}
    )
    explain_report(processed, llm1)
    assert len(llm1.calls) == 1

    clear_llm_cache()

    llm2 = FakeLLMClient(
        response_or_fn={"tests": [{"name": "HbA1c", "what_it_measures": text_hba1c}]}
    )
    explain_report(processed, llm2)
    assert len(llm2.calls) == 1


def test_indic_languages_never_call_llm():
    """Languages hi and te still make zero LLM calls."""
    raw_extraction = extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf"))
    processed = process_report(raw_extraction)

    llm_hi = FakeLLMClient()
    exp_hi = explain_report(processed, llm_hi, language="hi")
    assert len(llm_hi.calls) == 0
    assert all(e.used_fallback for e in exp_hi.explanations)

    llm_te = FakeLLMClient()
    exp_te = explain_report(processed, llm_te, language="te")
    assert len(llm_te.calls) == 0
    assert all(e.used_fallback for e in exp_te.explanations)


def test_banned_word_for_one_test_falls_back_for_that_test_only():
    """An LLM answer with a banned word for one test falls back for that test only."""
    raw_extraction = extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf"))
    processed = process_report(raw_extraction)

    good_hba1c = (
        "HbA1c measures average blood sugar levels over recent months. "
        "Doctors order it to check blood sugar patterns over time."
    )
    bad_glucose = (
        "Fasting glucose checks for diabetes in patients. "
        "Doctors order it to assess health."
    )

    fake_llm = FakeLLMClient(
        response_or_fn={
            "tests": [
                {"name": "HbA1c", "what_it_measures": good_hba1c},
                {"name": "Fasting Glucose", "what_it_measures": bad_glucose},
            ]
        }
    )

    explanation = explain_report(processed, fake_llm)
    hba1c_exp = next(e for e in explanation.explanations if e.test_name == "HbA1c")
    glucose_exp = next(e for e in explanation.explanations if e.test_name == "Fasting Glucose")

    assert hba1c_exp.used_fallback is False
    assert hba1c_exp.what_it_measures == good_hba1c

    assert glucose_exp.used_fallback is True
    assert "diabetes" not in glucose_exp.what_it_measures
