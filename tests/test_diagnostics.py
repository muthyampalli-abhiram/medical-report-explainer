"""Tests for LLM diagnostics, GroqClient attributes, validation reason codes, and LAST_LLM_DIAGNOSTICS."""

from datetime import date
import json
import httpx
import pytest

from app.models.lab import ExtractionResponse, LabReport, LabResult, ResultFlag
from app.services.explainer import (
    LAST_LLM_DIAGNOSTICS,
    clear_llm_cache,
    explain_report,
    validate_llm_text_reason,
)
from app.services.llm_client import (
    FakeLLMClient,
    GroqClient,
    LLMRateLimitError,
    LLMResponseError,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    clear_llm_cache()
    yield
    clear_llm_cache()


def _make_sample_response() -> ExtractionResponse:
    return ExtractionResponse(
        report=LabReport(
            report_date=date(2026, 1, 15),
            lab_name="Diagnostic Test Lab",
            results=[
                LabResult(
                    test_name_raw="HbA1c",
                    test_name_canonical="HbA1c",
                    value=7.4,
                    unit="%",
                    flag=ResultFlag.HIGH,
                ),
                LabResult(
                    test_name_raw="Fasting Blood Sugar",
                    test_name_canonical="Fasting Glucose",
                    value=132.0,
                    unit="mg/dL",
                    flag=ResultFlag.HIGH,
                ),
            ],
        ),
        warnings=[],
    )


# --- 1. GroqClient attributes ---

def test_groq_client_attributes_after_good_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"tests": []}'},
                    }
                ],
                "usage": {"completion_tokens": 42},
            },
        )

    client = GroqClient(
        api_key="test-key",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )
    result = client.generate_json("System prompt", "User prompt")
    assert result == {"tests": []}
    assert client.last_status_code == 200
    assert client.last_raw_content == '{"tests": []}'
    assert client.last_finish_reason == "stop"
    assert client.last_completion_tokens == 42


def test_groq_client_attributes_after_invalid_json():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "not valid json here"},
                    }
                ],
                "usage": {"completion_tokens": 15},
            },
        )

    client = GroqClient(
        api_key="test-key",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(LLMResponseError):
        client.generate_json("System prompt", "User prompt")

    assert client.last_status_code == 200
    assert client.last_raw_content == "not valid json here"
    assert client.last_finish_reason == "stop"
    assert client.last_completion_tokens == 15


def test_groq_client_attributes_after_429():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"message": "Rate limit reached"}},
        )

    client = GroqClient(
        api_key="test-key",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(LLMRateLimitError):
        client.generate_json("System prompt", "User prompt")

    assert client.last_status_code == 429


# --- 2. validate_llm_text_reason ---

def test_validate_llm_text_reason_empty():
    assert validate_llm_text_reason("", "HbA1c") == "empty"
    assert validate_llm_text_reason("   ", "HbA1c") == "empty"


def test_validate_llm_text_reason_too_long():
    text = "Potassium measures minerals. Doctors order it to check blood fats. " * 8
    assert len(text) > 320
    assert validate_llm_text_reason(text, "Potassium") == "too_long"


def test_validate_llm_text_reason_one_sentence():
    text = "Potassium measures minerals in your bloodstream."
    assert validate_llm_text_reason(text, "Potassium") == "sentence_count"


def test_validate_llm_text_reason_digit():
    text = "Potassium measures blood levels over 3 months. Doctors order it to check your heart rhythm."
    assert validate_llm_text_reason(text, "Potassium") == "digit"


def test_validate_llm_text_reason_banned_word_high():
    text = "Fasting glucose measures high sugar. Doctors order it to check blood fats."
    assert validate_llm_text_reason(text, "Fasting Glucose") == "banned_word:high"


def test_validate_llm_text_reason_good_text():
    text = "Potassium measures a mineral in the blood. Doctors order it to check heart rhythm."
    assert validate_llm_text_reason(text, "Potassium") is None


# --- 3. LAST_LLM_DIAGNOSTICS ---

def test_last_llm_diagnostics_rate_limit_error():
    clear_llm_cache()
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "Rate limit exceeded"})

    client = GroqClient(
        api_key="test-key",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )
    resp = _make_sample_response()
    expl = explain_report(resp, client, language="en")
    assert all(e.used_fallback for e in expl.explanations)

    # Should have entries for both explainable tests with llm_error:LLMRateLimitError
    assert len(LAST_LLM_DIAGNOSTICS) == 2
    assert LAST_LLM_DIAGNOSTICS[0] == {"test": "HbA1c", "reason": "llm_error:LLMRateLimitError"}
    assert LAST_LLM_DIAGNOSTICS[1] == {"test": "Fasting Glucose", "reason": "llm_error:LLMRateLimitError"}


def test_last_llm_diagnostics_missing_in_response():
    clear_llm_cache()
    client = FakeLLMClient({
        "tests": [
            {
                "name": "HbA1c",
                "what_it_measures": "HbA1c measures average blood sugar over time. Doctors order it to check blood fats.",
            }
        ]
    })
    resp = _make_sample_response()
    expl = explain_report(resp, client, language="en")

    reasons = {d["test"]: d["reason"] for d in LAST_LLM_DIAGNOSTICS}
    assert reasons["HbA1c"] == "accepted"
    assert reasons["Fasting Glucose"] == "missing_in_response"


def test_last_llm_diagnostics_cache_hit():
    clear_llm_cache()
    client = FakeLLMClient({
        "tests": [
            {
                "name": "HbA1c",
                "what_it_measures": "HbA1c measures average blood sugar over time. Doctors order it to check blood fats.",
            },
            {
                "name": "Fasting Glucose",
                "what_it_measures": "Fasting glucose measures sugar in the blood. Doctors order it to check blood fats.",
            },
        ]
    })
    resp = _make_sample_response()
    explain_report(resp, client, language="en")
    assert all(d["reason"] == "accepted" for d in LAST_LLM_DIAGNOSTICS)

    # Second call should all be cache_hit
    explain_report(resp, client, language="en")
    assert len(LAST_LLM_DIAGNOSTICS) == 2
    assert all(d["reason"] == "cache_hit" for d in LAST_LLM_DIAGNOSTICS)


def test_last_llm_diagnostics_contains_no_report_values():
    clear_llm_cache()
    client = FakeLLMClient({
        "tests": [
            {
                "name": "HbA1c",
                "what_it_measures": "HbA1c measures average blood sugar over time. Doctors order it to check blood fats.",
            },
            {
                "name": "Fasting Glucose",
                "what_it_measures": "Fasting glucose measures sugar in the blood. Doctors order it to check blood fats.",
            },
        ]
    })
    resp = _make_sample_response()
    explain_report(resp, client, language="en")

    for entry in LAST_LLM_DIAGNOSTICS:
        assert set(entry.keys()) == {"test", "reason"}
        # Ensure no values, units, or flags leak into diagnostics
        for val in ("7.4", "132", "%", "mg/dL", "HIGH", "Diagnostic Test Lab"):
            assert val != entry["test"]
            assert val != entry["reason"]
