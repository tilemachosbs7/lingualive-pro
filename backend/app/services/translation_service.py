import asyncio
import httpx
import logging
import os
import time
from typing import Optional

from ..config import settings

logger = logging.getLogger(__name__)


class TranslationService:
    """Translation service with multiple backends: DeepL (primary), OpenAI (fallback).
    
    Optimized for Deepgram+DeepL combo with:
    - Source language hints (no auto-detect overhead)
    - Glossary support
    - Timeout budgets
    - Latency-optimized DeepL model
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
        
        # Rate limiting state
        self._last_translate_time: float = 0.0
        self._rate_limit_ms = settings.translation_rate_limit_ms

    async def translate_text(
        self,
        text: str,
        source_lang: Optional[str],
        target_lang: str,
        provider: str = "deepl",
        timeout_ms: Optional[int] = None,
    ) -> str:
        """Translate using selected provider with timeout budget.
        
        Args:
            text: Text to translate
            source_lang: Source language code (None for auto-detect)
            target_lang: Target language code
            provider: 'deepl', 'openai', or 'google'
            timeout_ms: Optional timeout override in milliseconds
        
        Returns:
            Translated text
        """
        timeout_s = (timeout_ms or settings.translation_timeout_ms) / 1000.0
        
        try:
            return await asyncio.wait_for(
                self._translate_internal(text, source_lang, target_lang, provider),
                timeout=timeout_s
            )
        except asyncio.TimeoutError:
            logger.warning(f"Translation timeout after {timeout_s}s for provider={provider}")
            raise TimeoutError(f"Translation timed out after {timeout_s}s")
        except asyncio.CancelledError:
            logger.debug("Translation cancelled (newer request arrived)")
            raise

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
