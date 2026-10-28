"""LLM client interface and implementations for plain-language explanations."""

from abc import ABC, abstractmethod
import json
from typing import Any, Callable

import httpx
from app.config import settings


class LLMConfigError(Exception):
    """Raised when LLM configuration (such as API key or model) is missing or invalid."""

    pass


class LLMError(Exception):
    """Base exception for LLM service operational errors."""

    pass


class LLMResponseError(LLMError):
    """Raised when LLM response content or structure is invalid or unparseable."""

    pass


class LLMRateLimitError(LLMError):
    """Raised when LLM service returns HTTP 429 (rate limit exceeded)."""

    pass


class LLMClient(ABC):
    """Abstract base client for language model interactions."""

    @abstractmethod
    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Generate and parse a structured JSON response from system and user prompts."""
        pass


class GroqClient(LLMClient):
    """Groq client using OpenAI-compatible HTTP chat completions endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        raw_key = api_key if api_key is not None else settings.LLM_API_KEY
        raw_model = model if model is not None else settings.LLM_MODEL

        if not raw_key or not raw_key.strip():
            raise LLMConfigError(
                "Missing LLM_API_KEY. Set LLM_API_KEY in your environment or .env."
            )
        if not raw_model or not raw_model.strip():
            raise LLMConfigError(
                "Missing LLM_MODEL. Set LLM_MODEL in your environment or .env."
            )

        self.api_key = raw_key.strip()
        self.model = raw_model.strip()
        self.base_url = (base_url or "https://api.groq.com/openai/v1").rstrip("/")
        self.transport = transport
        self.provider = "groq"
        self.last_raw_content: str | None = None
        self.last_finish_reason: str | None = None
        self.last_completion_tokens: int | None = None
        self.last_status_code: int | None = None
        self.last_error_code: str | None = None

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Send chat completion request to Groq API with structured JSON output."""
        self.last_raw_content = None
        self.last_finish_reason = None
        self.last_completion_tokens = None
        self.last_status_code = None
        self.last_error_code = None

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "response_format": {"type": "json_object"},
        }

        if self.model.startswith("openai/gpt-oss"):
            reasoning_effort = getattr(settings, "LLM_REASONING_EFFORT", None)
            if reasoning_effort in ("low", "medium", "high"):
                payload["reasoning_effort"] = reasoning_effort

        client_kwargs: dict[str, Any] = {"timeout": 30.0}
        if self.transport is not None:
            client_kwargs["transport"] = self.transport

        try:
            with httpx.Client(**client_kwargs) as client:
                response = client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException:
            raise LLMError("LLM request timed out.")
        except httpx.RequestError:
            raise LLMError("LLM network request failed.")
        except Exception:
            raise LLMError("LLM request failed.")

        self.last_status_code = response.status_code
        data = None
        try:
            body_json = response.json()
            if isinstance(body_json, dict):
                data = body_json
                usage = data.get("usage")
                if isinstance(usage, dict) and "completion_tokens" in usage:
                    self.last_completion_tokens = usage.get("completion_tokens")

                choices = data.get("choices")
                if isinstance(choices, list) and len(choices) > 0 and isinstance(choices[0], dict):
                    choice = choices[0]
                    self.last_finish_reason = choice.get("finish_reason")
                    msg = choice.get("message")
                    if isinstance(msg, dict):
                        self.last_raw_content = msg.get("content")

                err = data.get("error")
                if isinstance(err, dict):
                    err_code = err.get("code")
                    if isinstance(err_code, str):
                        self.last_error_code = err_code
                    if self.last_completion_tokens is None:
                        err_usage = err.get("usage")
                        if isinstance(err_usage, dict) and "completion_tokens" in err_usage:
                            self.last_completion_tokens = err_usage.get("completion_tokens")
                        elif "completion_tokens" in err:
                            self.last_completion_tokens = err.get("completion_tokens")
                    if self.last_finish_reason is None and "finish_reason" in err:
                        self.last_finish_reason = err.get("finish_reason")
                elif isinstance(data.get("code"), str):
                    self.last_error_code = data.get("code")
        except Exception:
            pass

        if response.status_code == 429:
            raise LLMRateLimitError("The LLM service returned HTTP 429.")
        if response.status_code != 200:
            raise LLMError(f"The LLM service returned HTTP {response.status_code}.")

        if data is None:
            try:
                data = response.json()
            except Exception:
                raise LLMResponseError("Invalid JSON response from LLM service.")

        if not isinstance(data, dict):
            raise LLMResponseError("Unexpected response shape: body is not a JSON object.")

        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            raise LLMResponseError("Unexpected response shape: missing or empty choices.")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise LLMResponseError("Unexpected response shape: invalid choice entry.")

        finish_reason = first_choice.get("finish_reason")
        if self.last_finish_reason is None and finish_reason is not None:
            self.last_finish_reason = finish_reason

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise LLMResponseError("Unexpected response shape: missing message object.")

        content = message.get("content")
        if self.last_raw_content is None and content is not None:
            self.last_raw_content = content

        if (
            finish_reason == "length"
            or self.last_finish_reason == "length"
            or content is None
            or (isinstance(content, str) and not content.strip())
        ):
            raise LLMResponseError("The model ran out of tokens before answering.")

        if not isinstance(content, str):
            raise LLMResponseError("Unexpected response shape: message content is not a string.")

        try:
            parsed = json.loads(content)
        except Exception:
            raise LLMResponseError("Invalid JSON in LLM response message content.")

        if not isinstance(parsed, dict):
            raise LLMResponseError("LLM response message content is not a JSON object.")

        return parsed


class GeminiClient(LLMClient):
    """Google Gemini client using google-genai SDK."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key if api_key is not None else settings.LLM_API_KEY
        self.model = model if model is not None else settings.LLM_MODEL

        if not self.api_key or not self.api_key.strip():
            raise LLMConfigError(
                "Missing LLM_API_KEY. Set LLM_API_KEY in your environment or .env."
            )
        if not self.model or not self.model.strip():
            raise LLMConfigError(
                "Missing LLM_MODEL. Set LLM_MODEL in your environment or .env."
            )

        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(
            api_key=self.api_key.strip(),
            http_options=types.HttpOptions(timeout=30000),
        )

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Request structured JSON response with temperature 0.2 and 30s timeout."""
        config = self._types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
            response_mime_type="application/json",
            http_options=self._types.HttpOptions(timeout=30000),
        )
        response = self._client.models.generate_content(
            model=self.model.strip(),
            contents=user_prompt,
            config=config,
        )
        text = response.text or "{}"
        return json.loads(text)


class FakeLLMClient(LLMClient):
    """Mock LLM client recording all prompt invocations for testing."""

    def __init__(
        self,
        response_or_fn: dict[str, Any] | Callable[[str, str], dict[str, Any]] | None = None,
    ):
        self.calls: list[dict[str, str]] = []
        self._response_or_fn = (
            response_or_fn if response_or_fn is not None else {"tests": []}
        )

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if callable(self._response_or_fn):
            return self._response_or_fn(system_prompt, user_prompt)
        return self._response_or_fn


class UnavailableLLMClient(LLMClient):
    """Fallback LLM client that always raises, forcing use of catalog fallback descriptions."""

    def __init__(self, message: str = "LLM client unavailable"):
        self.message = message

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raise RuntimeError(self.message)


def get_llm_client(provider: str | None = None) -> LLMClient:
    """Factory creating an LLMClient instance based on provider string or settings."""
    active_provider = (provider or settings.LLM_PROVIDER or "gemini").lower().strip()
    if active_provider == "fake":
        return FakeLLMClient()
    elif active_provider == "gemini":
        return GeminiClient()
    elif active_provider == "groq":
        return GroqClient()
    else:
        raise ValueError(f"Unsupported LLM provider: {active_provider}")
