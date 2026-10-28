"""Tests for LLM client abstractions, configuration validation, and factory."""

import os
import pytest

from app.services.llm_client import (
    FakeLLMClient,
    GeminiClient,
    LLMConfigError,
    get_llm_client,
)


def test_factory_returns_fake_client():
    """Verify factory returns FakeLLMClient when 'fake' provider is specified."""
    client = get_llm_client("fake")
    assert isinstance(client, FakeLLMClient)


def test_gemini_client_missing_key_or_model_raises_config_error():
    """Verify GeminiClient raises LLMConfigError if api_key or model is empty/missing."""
    with pytest.raises(LLMConfigError, match="Missing LLM_API_KEY"):
        GeminiClient(api_key="", model="gemini-2.5-flash")

    with pytest.raises(LLMConfigError, match="Missing LLM_MODEL"):
        GeminiClient(api_key="fake-api-key", model="")


@pytest.mark.skipif(
    not os.environ.get("LLM_API_KEY") or not os.environ.get("LLM_MODEL"),
    reason="Live Gemini test requires LLM_API_KEY and LLM_MODEL set in environment",
)
def test_live_gemini_client():
    """Optional live integration test validating generate_json returns a dict."""
    client = GeminiClient(
        api_key=os.environ["LLM_API_KEY"],
        model=os.environ["LLM_MODEL"],
    )
    result = client.generate_json(
        system_prompt="Return JSON: {\"status\": \"ok\"}",
        user_prompt="Ping",
    )
    assert isinstance(result, dict)
    assert result.get("status") == "ok"
