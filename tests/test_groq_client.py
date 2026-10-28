"""Tests for GroqClient LLM provider integration, configuration, error handling, and privacy."""

import json
import os
from pathlib import Path
import pytest
import httpx

from app.config import settings
from app.services.explainer import clear_llm_cache, explain_report
from app.services.llm_client import (
    FakeLLMClient,
    GroqClient,
    LLMConfigError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    get_llm_client,
)
from app.services.pdf_extractor import extract_report_from_pdf
from app.services.pipeline import process_report

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def test_groq_client_mocked_200_returns_dict():
    """Verify that a mocked 200 response with message content JSON returns the parsed dict."""
    expected_content = '{"tests": [{"name": "HbA1c", "what_it_measures": "x"}]}'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": expected_content}}]},
        )

    client = GroqClient(
        api_key="test-key-123",
        model="llama-3.3-70b-versatile",
        transport=httpx.MockTransport(handler),
    )
    res = client.generate_json("System prompt", "User prompt")
    assert res == {"tests": [{"name": "HbA1c", "what_it_measures": "x"}]}


def test_groq_client_outgoing_request_structure():
    """Verify outgoing request path, headers, and payload structure for Groq API."""
    captured_request: dict[str, any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["auth_header"] = request.headers.get("Authorization")
        captured_request["content_type"] = request.headers.get("Content-Type")
        captured_request["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"tests": []}'}}]},
        )

    client = GroqClient(
        api_key="test-key-123",
        model="llama-3.3-70b-versatile",
        transport=httpx.MockTransport(handler),
    )
    client.generate_json("System instructions", "User query")

    assert captured_request["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured_request["auth_header"] == "Bearer test-key-123"
    assert "application/json" in captured_request["content_type"]

    body = captured_request["body"]
    assert body["model"] == "llama-3.3-70b-versatile"
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 4096
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"] == [
        {"role": "system", "content": "System instructions"},
        {"role": "user", "content": "User query"},
    ]


def test_groq_client_missing_key_or_model_raises_config_error():
    """Verify missing/empty key or model raises LLMConfigError and exception text never contains any key."""
    with pytest.raises(LLMConfigError) as exc_info1:
        GroqClient(api_key="", model="llama-3.3-70b-versatile")
    assert "test-key-123" not in str(exc_info1.value)

    with pytest.raises(LLMConfigError) as exc_info2:
        GroqClient(api_key="test-key-123", model="")
    assert "test-key-123" not in str(exc_info2.value)

    with pytest.raises(LLMConfigError) as exc_info3:
        GroqClient(api_key="   ", model="llama-3.3-70b-versatile")
    assert "test-key-123" not in str(exc_info3.value)

    with pytest.raises(LLMConfigError) as exc_info4:
        GroqClient(api_key="test-key-123", model="   ")
    assert "test-key-123" not in str(exc_info4.value)


@pytest.mark.parametrize(
    "status_code,response_text,expected_exc",
    [
        (401, '{"error": {"message": "Invalid API key - secret-response-body"}}', LLMError),
        (500, "Internal Server Error - secret-response-body", LLMError),
        (429, '{"error": {"message": "Rate limit exceeded - secret-response-body"}}', LLMRateLimitError),
    ],
)
def test_groq_client_http_errors(status_code: int, response_text: str, expected_exc: type[Exception]):
    """Verify HTTP 401, 500 raise LLMError and 429 raises LLMRateLimitError without leaking sensitive data."""
    secret_key = "test-key-123"
    secret_sys_prompt = "secret-system-prompt-abc"
    secret_user_prompt = "secret-user-prompt-xyz"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text=response_text)

    client = GroqClient(
        api_key=secret_key,
        model="llama-3.3-70b-versatile",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(expected_exc) as exc_info:
        client.generate_json(secret_sys_prompt, secret_user_prompt)

    exc_msg = str(exc_info.value)
    assert secret_key not in exc_msg
    assert secret_sys_prompt not in exc_msg
    assert secret_user_prompt not in exc_msg
    assert "secret-response-body" not in exc_msg


def test_groq_client_timeout_raises_llm_error():
    """Verify network timeouts raise LLMError without leaking key, prompts, or sensitive details."""
    secret_key = "test-key-123"
    secret_sys_prompt = "secret-system-prompt-abc"
    secret_user_prompt = "secret-user-prompt-xyz"

    def handler(request: httpx.Request):
        raise httpx.TimeoutException("Connection timed out - secret-network-timeout-body")

    client = GroqClient(
        api_key=secret_key,
        model="llama-3.3-70b-versatile",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMError) as exc_info:
        client.generate_json(secret_sys_prompt, secret_user_prompt)

    exc_msg = str(exc_info.value)
    assert secret_key not in exc_msg
    assert secret_sys_prompt not in exc_msg
    assert secret_user_prompt not in exc_msg
    assert "secret-network-timeout-body" not in exc_msg


def test_groq_client_invalid_content_raises_response_error():
    """Verify content not JSON, JSON list, or missing choices raises LLMResponseError."""
    # 1. Non-JSON content string
    def handler_non_json(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Not a JSON string."}}]},
        )

    client1 = GroqClient(
        api_key="test-key-123",
        model="llama-3.3-70b-versatile",
        transport=httpx.MockTransport(handler_non_json),
    )
    with pytest.raises(LLMResponseError):
        client1.generate_json("sys", "user")

    # 2. JSON list instead of JSON object
    def handler_json_list(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '[{"name": "HbA1c"}]'}}]},
        )

    client2 = GroqClient(
        api_key="test-key-123",
        model="llama-3.3-70b-versatile",
        transport=httpx.MockTransport(handler_json_list),
    )
    with pytest.raises(LLMResponseError):
        client2.generate_json("sys", "user")

    # 3. Missing choices in response body
    def handler_missing_choices(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"error": "Missing choices structure"},
        )

    client3 = GroqClient(
        api_key="test-key-123",
        model="llama-3.3-70b-versatile",
        transport=httpx.MockTransport(handler_missing_choices),
    )
    with pytest.raises(LLMResponseError):
        client3.generate_json("sys", "user")


def test_factory_returns_groq_client_and_fake(monkeypatch):
    """Verify factory returns GroqClient with provider 'groq' and FakeLLMClient with 'fake'."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "LLM_API_KEY", "test-key-123")
    monkeypatch.setattr(settings, "LLM_MODEL", "llama-3.3-70b-versatile")

    # Via settings
    client_default = get_llm_client()
    assert isinstance(client_default, GroqClient)

    # Via explicit parameter
    client_groq = get_llm_client("groq")
    assert isinstance(client_groq, GroqClient)

    # 'fake' still returns FakeLLMClient
    client_fake = get_llm_client("fake")
    assert isinstance(client_fake, FakeLLMClient)


def test_explain_report_integration_unsafe_and_429():
    """Verify unsafe response triggers fallback for HbA1c with no unsafe leakage, and 429 triggers clean fallback."""
    raw_extraction = extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf"))
    processed = process_report(raw_extraction)

    # 1. Unsafe diagnosis response
    def unsafe_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "tests": [
                                        {
                                            "name": "HbA1c",
                                            "what_it_measures": "You have diabetes.",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            },
        )

    unsafe_client = GroqClient(
        api_key="test-key-123",
        model="llama-3.3-70b-versatile",
        transport=httpx.MockTransport(unsafe_handler),
    )
    explanation = explain_report(processed, llm=unsafe_client)

    hba1c_exp = next(e for e in explanation.explanations if e.test_name == "HbA1c")
    assert hba1c_exp.used_fallback is True

    # The unsafe diagnosis text must appear nowhere in the output
    all_output_text = " ".join(
        [e.what_it_measures + " " + e.result_sentence for e in explanation.explanations]
        + [explanation.disclaimer, explanation.overall_notice]
    )
    assert "you have diabetes" not in all_output_text.lower()
    assert "diabetes" not in all_output_text.lower()

    # 2. Mock transport returning 429
    def rate_limit_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="Rate limit exceeded")

    rate_limit_client = GroqClient(
        api_key="test-key-123",
        model="llama-3.3-70b-versatile",
        transport=httpx.MockTransport(rate_limit_handler),
    )
    explanation_429 = explain_report(processed, llm=rate_limit_client)
    assert all(e.used_fallback for e in explanation_429.explanations)


def test_prompt_privacy_with_groq_client():
    """Verify user prompt sent to Groq contains canonical test names and no report values (7.4, 132)."""
    raw_extraction = extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf"))
    processed = process_report(raw_extraction)

    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"tests": []})}}]},
        )

    client = GroqClient(
        api_key="test-key-123",
        model="llama-3.3-70b-versatile",
        transport=httpx.MockTransport(handler),
    )
    explain_report(processed, llm=client)

    assert len(captured_requests) == 1
    req_body = json.loads(captured_requests[0].content.decode("utf-8"))
    user_prompt = next(m["content"] for m in req_body["messages"] if m["role"] == "user")

    # Canonical test names must be present
    assert "HbA1c" in user_prompt
    assert "Fasting Glucose" in user_prompt

    # Numbers from report must be strictly absent
    assert "7.4" not in user_prompt
    assert "132" not in user_prompt


def test_deps_get_llm_with_groq_and_no_key_returns_unavailable_fallback(monkeypatch):
    """Verify with LLM_PROVIDER=groq and no key, deps.get_llm returns UnavailableLLMClient and explanations use catalog fallback."""
    from app.api.deps import get_llm
    from app.services.llm_client import UnavailableLLMClient

    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "LLM_API_KEY", "")
    monkeypatch.setattr(settings, "LLM_MODEL", "")

    client = get_llm()
    assert isinstance(client, UnavailableLLMClient)

    raw_extraction = extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf"))
    processed = process_report(raw_extraction)

    explanation = explain_report(processed, llm=client)
    assert all(e.used_fallback for e in explanation.explanations)


@pytest.mark.skipif(
    os.environ.get("LLM_PROVIDER") != "groq"
    or not os.environ.get("LLM_API_KEY")
    or not os.environ.get("LLM_MODEL"),
    reason="Live Groq test requires LLM_PROVIDER=groq, LLM_API_KEY, and LLM_MODEL set in environment",
)
def test_live_groq_client():
    """Optional live integration test validating generate_json returns a dict when environment is configured."""
    client = GroqClient(
        api_key=os.environ["LLM_API_KEY"],
        model=os.environ["LLM_MODEL"],
    )
    result = client.generate_json(
        system_prompt='Return JSON: {"status": "ok"}',
        user_prompt="Ping",
    )
    assert isinstance(result, dict)
    assert result.get("status") == "ok"


def test_groq_client_payload_reasoning_effort_gpt_oss():
    """Verify request body contains max_tokens equal to settings.LLM_MAX_TOKENS and reasoning_effort 'low' for model openai/gpt-oss-20b."""
    captured_request: dict[str, any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"tests": []}'}}]},
        )

    client = GroqClient(
        api_key="test-key-123",
        model="openai/gpt-oss-20b",
        transport=httpx.MockTransport(handler),
    )
    client.generate_json("System instructions", "User query")

    body = captured_request["body"]
    assert body["max_tokens"] == settings.LLM_MAX_TOKENS
    assert body["reasoning_effort"] == "low"


def test_groq_client_payload_reasoning_effort_absent_for_other_model():
    """Verify reasoning_effort is absent for a model named qwen/qwen3.8-27b."""
    captured_request: dict[str, any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"tests": []}'}}]},
        )

    client = GroqClient(
        api_key="test-key-123",
        model="qwen/qwen3.8-27b",
        transport=httpx.MockTransport(handler),
    )
    client.generate_json("System instructions", "User query")

    body = captured_request["body"]
    assert "reasoning_effort" not in body


def test_groq_client_payload_invalid_reasoning_effort_omitted(monkeypatch):
    """Verify an invalid LLM_REASONING_EFFORT value ('extreme') is omitted."""
    monkeypatch.setattr(settings, "LLM_REASONING_EFFORT", "extreme")
    captured_request: dict[str, any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"tests": []}'}}]},
        )

    client = GroqClient(
        api_key="test-key-123",
        model="openai/gpt-oss-20b",
        transport=httpx.MockTransport(handler),
    )
    client.generate_json("System instructions", "User query")

    body = captured_request["body"]
    assert "reasoning_effort" not in body


def test_groq_client_finish_reason_length_and_empty_content_fallback():
    """Verify mocked response with finish_reason 'length' and empty content raises LLMResponseError without leaking key, and explain_report falls back cleanly."""
    secret_key = "secret-groq-key-length-123"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": ""},
                    }
                ],
                "usage": {"completion_tokens": 1024},
            },
        )

    client = GroqClient(
        api_key=secret_key,
        model="openai/gpt-oss-20b",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMResponseError) as exc_info:
        client.generate_json("System prompt", "User prompt")

    assert secret_key not in str(exc_info.value)
    assert client.last_finish_reason == "length"
    assert client.last_completion_tokens == 1024

    # explain_report falls back cleanly for every test
    clear_llm_cache()
    raw_extraction = extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf"))
    processed = process_report(raw_extraction)
    explanation = explain_report(processed, llm=client)
    assert len(explanation.explanations) > 0
    assert all(e.used_fallback for e in explanation.explanations)


def test_groq_client_mocked_400_json_validate_failed_fallback():
    """Verify mocked 400 with error code json_validate_failed sets last_error_code, still raises LLMError, and explain_report falls back cleanly."""
    secret_key = "secret-groq-key-400-xyz"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": "json_validate_failed",
                    "message": "Failed to validate JSON: secret-schema-failure",
                }
            },
        )

    client = GroqClient(
        api_key=secret_key,
        model="openai/gpt-oss-20b",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMError) as exc_info:
        client.generate_json("System prompt", "User prompt")

    assert secret_key not in str(exc_info.value)
    assert "secret-schema-failure" not in str(exc_info.value)
    assert client.last_status_code == 400
    assert client.last_error_code == "json_validate_failed"

    # explain_report falls back cleanly for every test
    clear_llm_cache()
    raw_extraction = extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf"))
    processed = process_report(raw_extraction)
    explanation = explain_report(processed, llm=client)
    assert len(explanation.explanations) > 0
    assert all(e.used_fallback for e in explanation.explanations)

