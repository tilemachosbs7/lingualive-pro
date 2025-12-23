import os
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load .env file BEFORE importing settings to ensure env vars are available
# When running from backend/ directory
env_path = Path(__file__).parent.parent / ".env"
if not env_path.exists():
    # When running from root directory
    env_path = Path(__file__).parent.parent.parent / "backend" / ".env"
    
load_dotenv(dotenv_path=env_path, override=True)
print(f"[STARTUP] Loading .env from: {env_path}")
print(f"[STARTUP] DEEPGRAM_API_KEY present: {bool(os.getenv('DEEPGRAM_API_KEY'))}")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.audio import audio_router
from .api.translate import router as translate_router
from .api.realtime import router as realtime_router
from .api.deepgram_realtime import router as deepgram_router
from .api.google_realtime import router as google_router
from .api.assemblyai_realtime import router as assemblyai_router
from .config import settings
from .services.translation_service import translation_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    yield
    # Shutdown - close HTTP clients
    await translation_service.close()


def create_app() -> FastAPI:
    app = FastAPI(title="LinguaLive v2", version="0.2.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(translate_router, prefix="/api")
    app.include_router(audio_router, prefix="/api")
    app.include_router(realtime_router)
    app.include_router(deepgram_router)  # TRUE real-time streaming
    app.include_router(google_router)    # Google Cloud Speech-to-Text
    app.include_router(assemblyai_router)  # AssemblyAI streaming STT

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
