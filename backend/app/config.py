from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI is optional - only required if using OpenAI provider
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")  # Optional for Deepgram+DeepL
    openai_translation_model: str = Field("gpt-4o-mini", alias="OPENAI_TRANSLATION_MODEL")
    openai_asr_model: str = Field("whisper-1", alias="OPENAI_ASR_MODEL")
    backend_port: int = Field(8000, alias="BACKEND_PORT")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], alias="CORS_ORIGINS")
    
    # Deepgram optimization settings
    deepgram_model: str = Field("nova-3", alias="DEEPGRAM_MODEL")  # nova-3 (latest) or nova-2
    deepgram_utterance_end_ms: int = Field(400, alias="DEEPGRAM_UTTERANCE_END_MS")  # Fallback if not sent by client
    deepgram_utterance_end_fast_ms: int = Field(280, alias="DEEPGRAM_UTTERANCE_END_FAST_MS")  # Fast mode: quicker sentence end
    deepgram_utterance_end_quality_ms: int = Field(600, alias="DEEPGRAM_UTTERANCE_END_QUALITY_MS")  # Quality: better coherence
    deepgram_vad_events: bool = Field(True, alias="DEEPGRAM_VAD_EVENTS")
    
    # Partial translation settings (live preview)
    partial_translate_interval_ms: int = Field(250, alias="PARTIAL_TRANSLATE_INTERVAL_MS")  # Faster partial translation (was 400)
    partial_translate_min_chars: int = Field(10, alias="PARTIAL_TRANSLATE_MIN_CHARS")  # Lower threshold (was 20)
    
    # Backpressure settings
    backpressure_p95_threshold_ms: int = Field(800, alias="BACKPRESSURE_P95_THRESHOLD_MS")  # If p95 > this, engage backpressure
    backpressure_min_chars_multiplier: float = Field(1.5, alias="BACKPRESSURE_MIN_CHARS_MULTIPLIER")  # Increase min chars by this
    
    # Translation optimization settings
    enable_syntax_fix: bool = Field(False, alias="ENABLE_SYNTAX_FIX")
    enable_two_pass_translation: bool = Field(True, alias="ENABLE_TWO_PASS_TRANSLATION")
    translation_timeout_ms: int = Field(5000, alias="TRANSLATION_TIMEOUT_MS")
    # AAA: Separate fast pass timeout (lower for responsiveness)
    translation_timeout_fast_ms: int = Field(2000, alias="TRANSLATION_TIMEOUT_FAST_MS")
    syntax_fix_timeout_ms: int = Field(3000, alias="SYNTAX_FIX_TIMEOUT_MS")
    translation_rate_limit_ms: int = Field(100, alias="TRANSLATION_RATE_LIMIT_MS")  # Reduced for faster response
    
    # DeepL specific
    deepl_glossary_id: str = Field("", alias="DEEPL_GLOSSARY_ID")
    deepl_formality: str = Field("default", alias="DEEPL_FORMALITY")  # default|more|less|prefer_more|prefer_less
    # AAA: DeepL fast/quality mode tuning
    deepl_fast_split_sentences: str = Field("0", alias="DEEPL_FAST_SPLIT_SENTENCES")  # "0" for single sentence, "1" for newlines
    deepl_quality_split_sentences: str = Field("nonewlines", alias="DEEPL_QUALITY_SPLIT_SENTENCES")  # More careful splitting
    
    # Caching & buffering
    enable_translation_cache: bool = Field(True, alias="ENABLE_TRANSLATION_CACHE")
    translation_cache_ttl_seconds: int = Field(3600, alias="TRANSLATION_CACHE_TTL_SECONDS")
    min_partial_chars: int = Field(15, alias="MIN_PARTIAL_CHARS")  # Only translate partials > N chars
    context_buffer_size: int = Field(3, alias="CONTEXT_BUFFER_SIZE")  # Keep N recent sentences for context
    
    # Advanced DeepL optimization
    deepl_retry_count: int = Field(2, alias="DEEPL_RETRY_COUNT")  # Retry on failure
    deepl_retry_delay_ms: int = Field(250, alias="DEEPL_RETRY_DELAY_MS")  # Base delay for 429 backoff (250->500->1000ms)
    enable_context_aware_translation: bool = Field(True, alias="ENABLE_CONTEXT_AWARE_TRANSLATION")
    
    # Deepgram confidence filtering
    min_confidence_threshold: float = Field(0.7, alias="MIN_CONFIDENCE_THRESHOLD")  # Skip low-confidence results
    min_confidence_threshold_speed: float = Field(0.5, alias="MIN_CONFIDENCE_THRESHOLD_SPEED")  # x2: Lower threshold in Speed Mode
    enable_confidence_filter: bool = Field(True, alias="ENABLE_CONFIDENCE_FILTER")
    
    # AAA: Adaptive syntax fix threshold (only run syntax fix if confidence < this)
    adaptive_syntax_threshold: float = Field(0.85, alias="ADAPTIVE_SYNTAX_THRESHOLD")
    
    # Smart partial handling
    min_words_for_translation: int = Field(3, alias="MIN_WORDS_FOR_TRANSLATION")  # Min words before translating
    max_words_before_flush: int = Field(15, alias="MAX_WORDS_BEFORE_FLUSH")  # AAA: Auto-flush after N words
    max_cache_size: int = Field(500, alias="MAX_CACHE_SIZE")  # Max cached translations
    
    # === AAA STUDIO ENHANCEMENTS ===
    
    # Filler word removal
    enable_filler_removal: bool = Field(True, alias="ENABLE_FILLER_REMOVAL")
    
    # Named entity preservation (DISABLED - causes placeholder artifacts)
    enable_entity_preservation: bool = Field(False, alias="ENABLE_ENTITY_PRESERVATION")
    
    # Technical term preservation (DISABLED - causes placeholder artifacts)
    enable_technical_preservation: bool = Field(False, alias="ENABLE_TECHNICAL_PRESERVATION")
    
    # Multi-language auto-detection
    enable_language_detection: bool = Field(True, alias="ENABLE_LANGUAGE_DETECTION")
    
    # Provider fallback chain
    enable_fallback_chain: bool = Field(True, alias="ENABLE_FALLBACK_CHAIN")
    fallback_chain: str = Field("deepl,openai,google", alias="FALLBACK_CHAIN")  # Comma-separated
    
    # Circuit breaker
    enable_circuit_breaker: bool = Field(True, alias="ENABLE_CIRCUIT_BREAKER")
    circuit_breaker_threshold: int = Field(5, alias="CIRCUIT_BREAKER_THRESHOLD")  # Failures before unhealthy
    circuit_breaker_cooldown_s: int = Field(30, alias="CIRCUIT_BREAKER_COOLDOWN_S")  # Seconds before retry
    
    # Dynamic rate limiting
    enable_dynamic_rate_limit: bool = Field(True, alias="ENABLE_DYNAMIC_RATE_LIMIT")
    base_rate_limit_ms: int = Field(100, alias="BASE_RATE_LIMIT_MS")
    
    # Quality scoring
    enable_quality_scoring: bool = Field(True, alias="ENABLE_QUALITY_SCORING")
    min_quality_score: float = Field(0.5, alias="MIN_QUALITY_SCORE")  # Warn if below
    
    # Metrics collection
    enable_metrics: bool = Field(True, alias="ENABLE_METRICS")
    
    # Glossary support
    enable_glossary: bool = Field(True, alias="ENABLE_GLOSSARY")
    glossary_domain: str = Field("", alias="GLOSSARY_DOMAIN")  # Domain for glossary lookup
    
    # Response streaming (word by word)
    enable_response_streaming: bool = Field(False, alias="ENABLE_RESPONSE_STREAMING")
    
    # Alternative translations
    enable_alternatives: bool = Field(False, alias="ENABLE_ALTERNATIVES")
    num_alternatives: int = Field(2, alias="NUM_ALTERNATIVES")
    
    # Confidence visualization
    enable_confidence_display: bool = Field(True, alias="ENABLE_CONFIDENCE_DISPLAY")
    
    # === ADVANCED OPTIMIZATIONS ===
    
    # Translation Memory (TM) with fuzzy matching
    enable_translation_memory: bool = Field(True, alias="ENABLE_TRANSLATION_MEMORY")
    tm_fuzzy_threshold: float = Field(0.85, alias="TM_FUZZY_THRESHOLD")  # Minimum similarity for fuzzy match
    tm_max_size: int = Field(10000, alias="TM_MAX_SIZE")  # Max TM entries
    
    # Context-Aware Translation
    enable_context_aware: bool = Field(True, alias="ENABLE_CONTEXT_AWARE")
    context_window_size: int = Field(3, alias="CONTEXT_WINDOW_SIZE")  # Number of previous sentences
    
    # Adaptive Confidence System
    enable_adaptive_confidence: bool = Field(True, alias="ENABLE_ADAPTIVE_CONFIDENCE")
    adaptive_min_confidence: float = Field(0.3, alias="ADAPTIVE_MIN_CONFIDENCE")  # Absolute minimum for adaptive system
    
    # Smart Punctuation Detection
    enable_smart_punctuation: bool = Field(True, alias="ENABLE_SMART_PUNCTUATION")
    pause_threshold_ms: int = Field(800, alias="PAUSE_THRESHOLD_MS")  # Pause = sentence end
    
    # Language Pair Profiles
    enable_language_profiles: bool = Field(True, alias="ENABLE_LANGUAGE_PROFILES")
    
    # Predictive Pre-caching
    enable_predictive_cache: bool = Field(True, alias="ENABLE_PREDICTIVE_CACHE")
    
    # === ADVANCED TRANSLATION REFINEMENTS ===
    
    # Back-Translation Validation
    enable_back_translation: bool = Field(True, alias="ENABLE_BACK_TRANSLATION")
    back_translation_threshold: float = Field(0.75, alias="BACK_TRANSLATION_THRESHOLD")
    
    # Hybrid Translation (consensus)
    enable_hybrid_translation: bool = Field(False, alias="ENABLE_HYBRID_TRANSLATION")  # Expensive - use for critical content
    
    # Domain Detection & Adaptation
    enable_domain_detection: bool = Field(True, alias="ENABLE_DOMAIN_DETECTION")
    
    # Sentence Complexity Analysis
    enable_complexity_analysis: bool = Field(True, alias="ENABLE_COMPLEXITY_ANALYSIS")
    
    # User Correction Learning
    enable_correction_learning: bool = Field(True, alias="ENABLE_CORRECTION_LEARNING")


settings = Settings()
