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
    deepl_formality: str = Field("default", alias="DEEPL_FORMALITY")  # default|more|less|prefer_more|prefer_less
    
    # Caching & buffering
    enable_translation_cache: bool = Field(True, alias="ENABLE_TRANSLATION_CACHE")
    translation_cache_ttl_seconds: int = Field(3600, alias="TRANSLATION_CACHE_TTL_SECONDS")
    min_partial_chars: int = Field(15, alias="MIN_PARTIAL_CHARS")  # Only translate partials > N chars
    context_buffer_size: int = Field(3, alias="CONTEXT_BUFFER_SIZE")  # Keep N recent sentences for context
    
    # Advanced DeepL optimization
    deepl_retry_count: int = Field(2, alias="DEEPL_RETRY_COUNT")  # Retry on failure
    deepl_retry_delay_ms: int = Field(200, alias="DEEPL_RETRY_DELAY_MS")  # Delay between retries
    enable_context_aware_translation: bool = Field(True, alias="ENABLE_CONTEXT_AWARE_TRANSLATION")
    
    # Deepgram confidence filtering
    min_confidence_threshold: float = Field(0.7, alias="MIN_CONFIDENCE_THRESHOLD")  # Skip low-confidence results
    enable_confidence_filter: bool = Field(True, alias="ENABLE_CONFIDENCE_FILTER")
    
    # Smart partial handling
    min_words_for_translation: int = Field(3, alias="MIN_WORDS_FOR_TRANSLATION")  # Min words before translating
    max_cache_size: int = Field(500, alias="MAX_CACHE_SIZE")  # Max cached translations


settings = Settings()
