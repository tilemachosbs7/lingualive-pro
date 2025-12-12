import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from ..services.asr_service import ASRServiceError, asr_service
from ..services.translation_service import translation_service

logger = logging.getLogger(__name__)


audio_router = APIRouter(tags=["audio"])


class TranscribeAudioResponse(BaseModel):
    text: str = Field(..., description="Transcript text")
    detected_lang: Optional[str] = Field(None, description="Language detected by ASR provider")


class TranslateAudioResponse(BaseModel):
    original_text: str = Field(..., description="Original transcript")
    translated_text: str = Field(..., description="Translated transcript")
    source_lang: str = Field(..., description="Source language used")
    target_lang: str = Field(..., description="Target language requested")


@audio_router.post("/transcribe-audio", response_model=TranscribeAudioResponse)
async def transcribe_audio(file: UploadFile = File(...), source_lang: Optional[str] = Form(None)) -> TranscribeAudioResponse:
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio file is empty.")

    try:
        result = await asr_service.transcribe_audio(audio_bytes, source_lang=source_lang or None)
    except ASRServiceError as exc:
        logger.error("ASR transcription failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Transcription failed.") from exc

    return TranscribeAudioResponse(text=result.text, detected_lang=result.detected_lang)


@audio_router.post("/translate-audio", response_model=TranslateAudioResponse)
async def translate_audio(
    file: UploadFile = File(...),
    source_lang: str = Form("auto"),
    target_lang: str = Form(...),
) -> TranslateAudioResponse:
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio file is empty.")

    if not target_lang:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Target language is required.")

    detected_lang: Optional[str] = None
    try:
        asr_result = await asr_service.transcribe_audio(audio_bytes, source_lang=None if source_lang == "auto" else source_lang)
        detected_lang = asr_result.detected_lang
        effective_source = detected_lang or source_lang or "auto"

        translated_text = await translation_service.translate_text(
            text=asr_result.text,
            source_lang=effective_source if effective_source != "auto" else None,
            target_lang=target_lang,
        )
    except ASRServiceError as exc:
        logger.error("ASR transcription failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Translation failed, please try again.") from exc
    except ValueError as exc:
        logger.error("Translation validation failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - provider/network failures
        logger.exception("Audio translation failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Translation failed, please try again.") from exc

    return TranslateAudioResponse(
        original_text=asr_result.text,
        translated_text=translated_text,
        source_lang=detected_lang or source_lang,
        target_lang=target_lang,
    )
