import logging
from dataclasses import dataclass

import httpx

from ..config import settings

logger = logging.getLogger(__name__)


class ASRServiceError(Exception):
    """Represents errors returned by the ASR provider."""


@dataclass
class ASRResult:
    text: str
    detected_lang: str | None


class ASRService:
    """Simple ASR client using OpenAI audio transcription API."""

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required.")
        self.api_key = api_key
        self.model = model
        self._endpoint = "https://api.openai.com/v1/audio/transcriptions"

    async def transcribe_audio(self, audio_bytes: bytes, *, source_lang: str | None = None) -> ASRResult:
        if not audio_bytes:
            raise ASRServiceError("No audio data provided.")

        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Build multipart form data properly
        files = {
            "file": ("audio.webm", audio_bytes, "audio/webm"),
            "model": (None, self.model),
        }
        if source_lang and source_lang != "auto":
            files["language"] = (None, source_lang)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(self._endpoint, headers=headers, files=files)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            logger.error("ASR request failed: %s", exc.response.text)
            raise ASRServiceError("ASR provider error") from exc
        except Exception as exc:  # pragma: no cover - network/transport errors
            logger.exception("ASR request crashed")
            raise ASRServiceError("ASR request failed") from exc

        text = data.get("text")
        if not text or not isinstance(text, str):
            raise ASRServiceError("ASR provider returned empty transcript")

        detected_lang = data.get("language") if isinstance(data, dict) else None
        return ASRResult(text=text.strip(), detected_lang=detected_lang)


asr_service = ASRService(api_key=settings.openai_api_key, model=settings.openai_asr_model)
