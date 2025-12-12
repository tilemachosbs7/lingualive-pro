import httpx
import os

from ..config import settings


class TranslationService:
    """Translation service with multiple backends: DeepL (primary), OpenAI (fallback)."""

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required.")
        self.api_key = api_key
        self.model = model
        self._client = httpx.AsyncClient(timeout=10.0)
        
        # DeepL configuration
        self.deepl_key = os.getenv("DEEPL_API_KEY", "")
        self.deepl_endpoint = "https://api-free.deepl.com/v1/translate"  # Free tier
        if self.deepl_key.startswith("sk-"):  # Pro tier
            self.deepl_endpoint = "https://api.deepl.com/v1/translate"

    async def translate_text(self, text: str, source_lang: str | None, target_lang: str, provider: str = "deepl") -> str:
        """Translate using selected provider or fallback."""
        
        # Google Cloud Translation
        if provider == "google":
            try:
                return await self._translate_google(text, target_lang, source_lang)
            except Exception as e:
                print(f"Google Cloud error (falling back to OpenAI): {e}")
                return await self._translate_openai(text, target_lang)
        
        # DeepL
        if provider == "deepl" and self.deepl_key:
            try:
                return await self._translate_deepl(text, target_lang)
            except Exception as e:
                print(f"DeepL error (falling back to OpenAI): {e}")
                return await self._translate_openai(text, target_lang)
        
        # Fallback or selected OpenAI
        return await self._translate_openai(text, target_lang)

    async def _translate_deepl(self, text: str, target_lang: str) -> str:
        """Translate using DeepL API with latency-optimized settings."""
        # Map language codes to DeepL format
        lang_map = {
            "el": "EL",  # Greek
            "en": "EN-US",  # English
            "fr": "FR",  # French
            "de": "DE",  # German
            "es": "ES",  # Spanish
            "it": "IT",  # Italian
            "pt": "PT-BR",  # Portuguese
            "ja": "JA",  # Japanese
            "zh": "ZH",  # Chinese
        }
        
        target = lang_map.get(target_lang, target_lang.upper())
        
        payload = {
            "text": text,
            "target_lang": target,
            "model_type": "latency_optimized",  # FASTEST model
            "split_sentences": "0",  # Don't split - already one sentence
        }
        
        headers = {
            "Authorization": f"DeepL-Auth-Key {self.deepl_key}",
        }
        
        response = await self._client.post(
            self.deepl_endpoint,
            headers=headers,
            data=payload,
        )
        response.raise_for_status()
        data = response.json()
        
        translations = data.get("translations", [])
        if translations:
            return translations[0].get("text", "").strip()
        
        raise ValueError("DeepL returned no translations")

    async def _translate_openai(self, text: str, target_lang: str) -> str:
        """Translate using OpenAI (fallback)."""
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

        return content.strip()

    async def _translate_google(self, text: str, target_lang: str, source_lang: str | None = None) -> str:
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
            print(f"Google Cloud Translation error: {e}")
            raise


translation_service = TranslationService(
    api_key=settings.openai_api_key,
    model=settings.openai_translation_model,
)
