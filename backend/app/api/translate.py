from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ..services.translation_service import translation_service


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source_lang: str | None = None
    target_lang: str = Field(..., min_length=1)


class TranslateResponse(BaseModel):
    translated_text: str


router = APIRouter(tags=["translate"])


@router.post("/translate-text", response_model=TranslateResponse)
async def translate(payload: TranslateRequest) -> TranslateResponse:
    try:
        translated = await translation_service.translate_text(
            text=payload.text,
            source_lang=payload.source_lang,
            target_lang=payload.target_lang,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Translation provider error",
        ) from exc

    return TranslateResponse(translated_text=translated)
