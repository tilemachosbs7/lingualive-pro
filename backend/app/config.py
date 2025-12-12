from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_translation_model: str = Field("gpt-4o-mini", alias="OPENAI_TRANSLATION_MODEL")
    openai_asr_model: str = Field("whisper-1", alias="OPENAI_ASR_MODEL")
    backend_port: int = Field(8000, alias="BACKEND_PORT")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], alias="CORS_ORIGINS")


settings = Settings()
