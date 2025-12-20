from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from ..services.translation_service import translation_service
from ..services.enhancement_service import enhancement_service
from ..services.advanced_optimizations import optimization_controller
from ..services.advanced_translation_refinements import advanced_refinement_controller


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source_lang: str | None = None
    target_lang: str = Field(..., min_length=1)


class TranslateResponse(BaseModel):
    translated_text: str


class MetricsResponse(BaseModel):
    translation_metrics: Dict[str, Any]
    provider_health: Dict[str, Any]
    rate_limiter: Dict[str, Any]
    translation_service_stats: Dict[str, Any]
    optimization_stats: Dict[str, Any]


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


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """Get comprehensive translation metrics and provider health status."""
    enhancement_metrics = enhancement_service.get_metrics_summary()
    translation_stats = translation_service.get_stats()
    optimization_stats = optimization_controller.get_comprehensive_stats()
    
    return MetricsResponse(
        translation_metrics=enhancement_metrics.get("translation_metrics", {}),
        provider_health=enhancement_metrics.get("provider_health", {}),
        rate_limiter=enhancement_metrics.get("rate_limiter", {}),
        translation_service_stats=translation_stats,
        optimization_stats=optimization_stats,
    )


@router.post("/metrics/reset")
async def reset_metrics():
    """Reset all metrics counters."""
    # Reset enhancement service metrics
    enhancement_service.metrics = enhancement_service.metrics.__class__()
    return {"status": "ok", "message": "Metrics reset successfully"}
