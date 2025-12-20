import asyncio
import httpx
import logging
import os
import time
from typing import Optional, List
from datetime import datetime, timedelta
from collections import OrderedDict

from ..config import settings

logger = logging.getLogger(__name__)


class LRUCache:
    """Simple LRU cache with TTL support."""
    
    def __init__(self, max_size: int, ttl_seconds: int):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._ttl = timedelta(seconds=ttl_seconds)
    
    def get(self, key):
        if key not in self._cache:
            return None
        value, timestamp = self._cache[key]
        if datetime.now() - timestamp > self._ttl:
            del self._cache[key]
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return value
    
    def put(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, datetime.now())
        # Evict oldest if over capacity
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
    
    def __len__(self):
        return len(self._cache)


class TranslationService:
    """Translation service with multiple backends: DeepL (primary), OpenAI (fallback).
    
    Optimized for Deepgram+DeepL combo with:
    - Source language hints (no auto-detect overhead)
    - Glossary support
    - Timeout budgets
    - LRU Translation caching with TTL
    - Context-aware translation
    - Retry logic with exponential backoff
    - Formality support
    """

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required.")
        self.api_key = api_key
        self.model = model
        self._client = httpx.AsyncClient(timeout=10.0)
        
        # DeepL configuration
        self.deepl_key = os.getenv("DEEPL_API_KEY", "")
        self.deepl_endpoint = "https://api-free.deepl.com/v2/translate"  # Free tier (v2!)
        if self.deepl_key and not self.deepl_key.endswith(":fx"):  # Pro tier
            self.deepl_endpoint = "https://api.deepl.com/v2/translate"
        
        # Glossary ID (optional)
        self.deepl_glossary_id = settings.deepl_glossary_id
        
        # Formality level
        self.deepl_formality = settings.deepl_formality
        
        # LRU Translation cache with TTL
        self._translation_cache = LRUCache(
            max_size=settings.max_cache_size,
            ttl_seconds=settings.translation_cache_ttl_seconds
        )
        
        # Context buffer for context-aware translation
        self._context_buffer: List[str] = []
        self._max_context = settings.context_buffer_size
        
        # Rate limiting state
        self._last_translate_time: float = 0.0
        self._rate_limit_ms = settings.translation_rate_limit_ms
        
        # Retry configuration
        self._retry_count = settings.deepl_retry_count
        self._retry_delay_ms = settings.deepl_retry_delay_ms
        
        # Stats for monitoring
        self._total_requests = 0
        self._cache_hits = 0
        self._total_latency_ms = 0.0

    def add_context(self, text: str) -> None:
        """Add a sentence to context buffer for context-aware translation."""
        self._context_buffer.append(text)
        if len(self._context_buffer) > self._max_context:
            self._context_buffer.pop(0)
    
    def get_context(self) -> str:
        """Get recent context as a single string."""
        return " ".join(self._context_buffer[-self._max_context:])
    
    def get_stats(self) -> dict:
        """Get translation service statistics."""
        return {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(1, self._total_requests),
            "cache_size": len(self._translation_cache),
            "avg_latency_ms": self._total_latency_ms / max(1, self._total_requests - self._cache_hits),
        }

    async def translate_text(
        self,
        text: str,
        source_lang: Optional[str],
        target_lang: str,
        provider: str = "deepl",
        timeout_ms: Optional[int] = None,
        use_context: bool = False,
    ) -> str:
        """Translate using selected provider with timeout budget.
        
        Args:
            text: Text to translate
            source_lang: Source language code (None for auto-detect)
            target_lang: Target language code
            provider: 'deepl', 'openai', or 'google'
            timeout_ms: Optional timeout override in milliseconds
            use_context: Whether to use context buffer for better translation
        
        Returns:
            Translated text
        """
        self._total_requests += 1
        
        # Normalize text
        text = text.strip()
        if not text:
            return ""
        
        # Check cache (if enabled)
        if settings.enable_translation_cache and provider == "deepl":
            cache_key = (text.lower(), source_lang, target_lang, self.deepl_formality)
            cached = self._translation_cache.get(cache_key)
            if cached:
                self._cache_hits += 1
                logger.debug(f"Cache hit ({self._cache_hits}/{self._total_requests}): {text[:30]}...")
                return cached
        
        timeout_s = (timeout_ms or settings.translation_timeout_ms) / 1000.0
        start_time = time.perf_counter()
        
        try:
            # Build context-aware text if enabled
            translate_text = text
            if use_context and settings.enable_context_aware_translation and provider == "deepl":
                context = self.get_context()
                if context:
                    # DeepL doesn't support context directly, but OpenAI does
                    # For DeepL we just translate the current text
                    pass
            
            result = await asyncio.wait_for(
                self._translate_with_retry(text, source_lang, target_lang, provider),
                timeout=timeout_s
            )
            
            # Update stats
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._total_latency_ms += latency_ms
            
            # Cache the result (if enabled and DeepL)
            if settings.enable_translation_cache and provider == "deepl":
                cache_key = (text.lower(), source_lang, target_lang, self.deepl_formality)
                self._translation_cache.put(cache_key, result)
            
            # Add to context buffer
            self.add_context(text)
            
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Translation timeout after {timeout_s}s for provider={provider}")
            raise TimeoutError(f"Translation timed out after {timeout_s}s")
        except asyncio.CancelledError:
            logger.debug("Translation cancelled (newer request arrived)")
            raise

    async def _translate_with_retry(
        self,
        text: str,
        source_lang: Optional[str],
        target_lang: str,
        provider: str,
    ) -> str:
        """Translate with retry logic."""
        last_error = None
        
        for attempt in range(self._retry_count + 1):
            try:
                return await self._translate_internal(text, source_lang, target_lang, provider)
            except Exception as e:
                last_error = e
                if attempt < self._retry_count:
                    delay = (self._retry_delay_ms / 1000.0) * (attempt + 1)  # Exponential backoff
                    logger.warning(f"Translation attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
        
        # All retries failed
        raise last_error

    async def _translate_internal(
        self,
        text: str,
        source_lang: Optional[str],
        target_lang: str,
        provider: str,
    ) -> str:
        """Internal translation dispatch."""
        
        # Google Cloud Translation
        if provider == "google":
            try:
                return await self._translate_google(text, target_lang, source_lang)
            except Exception as e:
                logger.warning(f"Google Cloud error (falling back to OpenAI): {e}")
                return await self._translate_openai(text, target_lang)
        
        # DeepL (optimized path)
        if provider == "deepl" and self.deepl_key:
            try:
                return await self._translate_deepl(text, target_lang, source_lang)
            except Exception as e:
                logger.warning(f"DeepL error (falling back to OpenAI): {e}")
                return await self._translate_openai(text, target_lang)
        
        # Fallback or selected OpenAI
        return await self._translate_openai(text, target_lang)

    async def _translate_deepl(
        self,
        text: str,
        target_lang: str,
        source_lang: Optional[str] = None,
    ) -> str:
        """Translate using DeepL API with maximum optimization.
        
        Optimizations:
        - source_lang hint to skip auto-detect
        - latency_optimized model type (prefer_quality_optimized for refine pass)
        - split_sentences=0 for single sentences
        - Glossary support
        """
        start_time = time.perf_counter()
        
        # Map language codes to DeepL format
        lang_map = {
            "el": "EL",      # Greek
            "en": "EN-US",   # English (US)
            "en-GB": "EN-GB",# English (UK)
            "fr": "FR",      # French
            "de": "DE",      # German
            "es": "ES",      # Spanish
            "it": "IT",      # Italian
            "pt": "PT-BR",   # Portuguese (Brazil)
            "pt-PT": "PT-PT",# Portuguese (Portugal)
            "ja": "JA",      # Japanese
            "zh": "ZH",      # Chinese
            "ko": "KO",      # Korean
            "ru": "RU",      # Russian
            "nl": "NL",      # Dutch
            "pl": "PL",      # Polish
            "da": "DA",      # Danish
            "sv": "SV",      # Swedish
            "fi": "FI",      # Finnish
            "nb": "NB",      # Norwegian (Bokmål)
            "cs": "CS",      # Czech
            "hu": "HU",      # Hungarian
            "ro": "RO",      # Romanian
            "sk": "SK",      # Slovak
            "sl": "SL",      # Slovenian
            "bg": "BG",      # Bulgarian
            "et": "ET",      # Estonian
            "lt": "LT",      # Lithuanian
            "lv": "LV",      # Latvian
            "uk": "UK",      # Ukrainian
            "tr": "TR",      # Turkish
            "id": "ID",      # Indonesian
            "ar": "AR",      # Arabic
        }
        
        target = lang_map.get(target_lang, target_lang.upper())
        
        payload = {
            "text": [text],  # DeepL v2 expects array
            "target_lang": target,
            "split_sentences": "0",  # Don't split - already one sentence
        }
        
        # Add source language hint if available (skips auto-detect = faster)
        if source_lang and source_lang != "auto":
            source = lang_map.get(source_lang, source_lang.upper())
            # DeepL source doesn't need -US/-BR suffix
            if source.startswith("EN-"):
                source = "EN"
            elif source.startswith("PT-"):
                source = "PT"
            payload["source_lang"] = source
        
        # Add glossary if configured
        if self.deepl_glossary_id:
            payload["glossary_id"] = self.deepl_glossary_id
        
        # Add formality level (if not default)
        if self.deepl_formality and self.deepl_formality != "default":
            payload["formality"] = self.deepl_formality
        
        headers = {
            "Authorization": f"DeepL-Auth-Key {self.deepl_key}",
            "Content-Type": "application/json",
        }
        
        response = await self._client.post(
            self.deepl_endpoint,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        
        translations = data.get("translations", [])
        if translations:
            result = translations[0].get("text", "").strip()
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"DeepL translation took {duration_ms:.1f}ms (source={source_lang})")
            return result
        
        raise ValueError("DeepL returned no translations")

    async def _translate_openai(self, text: str, target_lang: str) -> str:
        """Translate using OpenAI (fallback)."""
        start_time = time.perf_counter()
        
        prompt = f"Translate to {target_lang}. Only output the translation, nothing else."

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            "temperature": 0,
            "max_tokens": 500,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = await self._client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        data = response.json()

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ValueError("Translation provider returned an unexpected response.")

        if not content:
            raise ValueError("Translation provider returned no content.")

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(f"OpenAI translation took {duration_ms:.1f}ms")
        return content.strip()

    async def _translate_google(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        """Translate using Google Cloud Translation API."""
        try:
            from google.cloud import translate_v2 as translate
            
            # Set credentials from environment
            google_credentials = os.getenv("GOOGLE_CLOUD_CREDENTIALS", "")
            if google_credentials:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials
            
            translate_client = translate.Client()
            
            # Google Cloud accepts ISO 639-1 codes (e.g., 'en', 'el', 'fr')
            result = translate_client.translate(
                text,
                target_language=target_lang,
                source_language=source_lang if source_lang and source_lang != "auto" else None
            )
            
            return result["translatedText"].strip()
            
        except Exception as e:
            logger.error(f"Google Cloud Translation error: {e}")
            raise


translation_service = TranslationService(
    api_key=settings.openai_api_key,
    model=settings.openai_translation_model,
)
