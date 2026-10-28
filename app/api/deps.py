"""FastAPI dependencies for database path and LLM client."""

from pathlib import Path

from app.config import settings
from app.services.llm_client import LLMClient, UnavailableLLMClient, get_llm_client


def get_db_path() -> Path:
    """Return configured SQLite database path."""
    return Path(settings.DB_PATH)


def get_llm() -> LLMClient:
    """Return active LLM client or UnavailableLLMClient fallback on error."""
    try:
        return get_llm_client()
    except Exception:
        return UnavailableLLMClient()
