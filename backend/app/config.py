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
    
    # Deepgram optimization settings
    deepgram_utterance_end_ms: int = Field(400, alias="DEEPGRAM_UTTERANCE_END_MS")
    deepgram_vad_events: bool = Field(True, alias="DEEPGRAM_VAD_EVENTS")
    
    # Translation optimization settings
    enable_syntax_fix: bool = Field(False, alias="ENABLE_SYNTAX_FIX")
    enable_two_pass_translation: bool = Field(True, alias="ENABLE_TWO_PASS_TRANSLATION")
    translation_timeout_ms: int = Field(5000, alias="TRANSLATION_TIMEOUT_MS")
    syntax_fix_timeout_ms: int = Field(3000, alias="SYNTAX_FIX_TIMEOUT_MS")
    translation_rate_limit_ms: int = Field(300, alias="TRANSLATION_RATE_LIMIT_MS")
    
    # DeepL specific
    deepl_glossary_id: str = Field("", alias="DEEPL_GLOSSARY_ID")


settings = Settings()
