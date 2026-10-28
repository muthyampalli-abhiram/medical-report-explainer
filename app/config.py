from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_ENV: str = "development"
    DB_PATH: str = "data/health.db"
    LLM_PROVIDER: str = "gemini"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = ""
    LLM_REASONING_EFFORT: str = "low"
    LLM_MAX_TOKENS: int = 4096
    MAX_UPLOAD_MB: int = 10
    TESSERACT_CMD: str = ""
    OCR_VERIFY_BELOW: float = 0.85
    OCR_UNKNOWN_BELOW: float = 0.60
    OCR_MAX_PAGES: int = 5
    OCR_MAX_IMAGE_PIXELS: int = 40000000


settings = Settings()
