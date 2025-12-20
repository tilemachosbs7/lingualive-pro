"""
WebSocket endpoint for real-time transcription using AssemblyAI v3.
Uses streaming recognition with interim results.
AssemblyAI streaming supports: en, es, fr, de, it, pt (and auto-detect)
"""

import asyncio
import base64
import json
import logging
import os
from urllib.parse import urlencode

import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import settings
from ..services.translation_service import translation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["assemblyai"])

# AssemblyAI streaming endpoint (v3) - can be overridden via env for EU endpoint
DEFAULT_ASSEMBLYAI_URL = "wss://streaming.assemblyai.com/v3/ws"

# AssemblyAI streaming supported explicit languages
SUPPORTED_LANGUAGES = {"en", "es", "fr", "de", "it", "pt", "auto"}


async def check_and_fix_syntax(text: str) -> str:
    """
    Quick syntax/grammar check using GPT.
    Only called in quality mode. Returns original text on any error.
    """
    import httpx
    
    openai_key = getattr(settings, 'openai_api_key', None) or os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        return text
    
    prompt = (
        "You are a transcription corrector. Fix any transcription errors, "
        "grammar issues, or punctuation problems in the following text. "
        "Output ONLY the corrected text with no explanation, no quotes, no prefix. "
        "If the text is already correct, output it unchanged."
    )
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "max_tokens": 500,
    }
    
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json",
    }
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"Syntax check failed (continuing with original): {e}")
        return text


def build_assemblyai_url(source_lang: str, sample_rate: int, quality_mode: str) -> str:
    """
    Build AssemblyAI v3 WebSocket URL with proper query parameters.
    
    Per official docs:
    - sample_rate: required (e.g., 16000)
    - encoding: pcm_s16le for raw PCM16
    - speech_model: universal-streaming-english or universal-streaming-multilingual
    - language_detection: true only when auto-detecting
    - format_turns: true for formatted output in quality mode
    """
    base_url = os.getenv("ASSEMBLYAI_STREAMING_BASE_URL", DEFAULT_ASSEMBLYAI_URL)
    
    params = {
        "sample_rate": sample_rate,
        "encoding": "pcm_s16le",
    }
    
    # Select speech model based on source language
    if source_lang == "en":
        params["speech_model"] = "universal-streaming-english"
    else:
        # For es/fr/de/it/pt or auto, use multilingual model
        params["speech_model"] = "universal-streaming-multilingual"
    
    # Enable language detection only for auto mode
    if source_lang == "auto":
        params["language_detection"] = "true"
    
    # Enable formatted turns in quality mode
    if quality_mode == "quality":
        params["format_turns"] = "true"
    else:
        params["format_turns"] = "false"
    
    return f"{base_url}?{urlencode(params)}"


@router.websocket("/assemblyai")
async def assemblyai_transcription(websocket: WebSocket) -> None:
    """
    Real-time transcription using AssemblyAI streaming API v3.
    Supports: en, es, fr, de, it, pt, and auto-detect.
    """
    await websocket.accept()
    logger.info("Client connected to AssemblyAI endpoint")

    # Get API key at runtime
    assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY", "")
    
    if not assemblyai_api_key:
        await websocket.send_json({
            "type": "error",
            "message": "ASSEMBLYAI_API_KEY not configured"
        })
        await websocket.close()
        return

    target_lang = "en"
    source_lang = "en"
    translation_provider = "deepl"
    quality_mode = "fast"  # fast or quality
    current_sentence = ""
    assemblyai_ws = None
    listen_task = None
    ws_closed = False
    sample_rate = 16000  # AssemblyAI default

    async def process_and_translate(text: str) -> None:
        """Process text (optionally fix syntax) and translate."""
        nonlocal ws_closed
        if ws_closed:
            return
        
        processed_text = text
        
        try:
            # Only run syntax check in quality mode
            if quality_mode == "quality":
                corrected = await check_and_fix_syntax(text)
                
                if corrected != text and not ws_closed:
                    try:
                        await websocket.send_json({
                            "type": "original_corrected",
                            "original": text,
                            "corrected": corrected,
                        })
                    except Exception:
                        pass
                    processed_text = corrected
            
            # Translate
            translation = await translation_service.translate_text(
                text=processed_text,
                target_lang=target_lang,
                source_lang=source_lang if source_lang != "auto" else None,
                provider=translation_provider,
            )
            
            if not ws_closed:
                try:
                    await websocket.send_json({
                        "type": "translation",
                        "original": processed_text,
                        "translation": translation,
                    })
                except Exception:
                    pass
            
        except Exception as e:
            logger.error(f"Process/translate error: {e}")

    async def listen_to_assemblyai() -> None:
        """Listen for AssemblyAI v3 responses."""
        nonlocal current_sentence, ws_closed
        try:
            async for message in assemblyai_ws:
                if ws_closed:
                    break
                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "")
                    
                    # Handle different message types from AssemblyAI v3
                    if msg_type == "Begin":
                        logger.info(f"AssemblyAI session started: {data.get('session_id')}")
                        
                    elif msg_type == "Turn":
                        # Turn contains transcript parts (v3 format)
                        transcript = data.get("transcript", "")
                        end_of_turn = data.get("end_of_turn", False)
                        
                        if transcript:
                            if end_of_turn:
                                # Final result for this turn
                                current_sentence += transcript + " "
                                final_text = current_sentence.strip()
                                current_sentence = ""
                                
                                try:
                                    await websocket.send_json({
                                        "type": "original_complete",
                                        "text": final_text,
                                    })
                                except Exception:
                                    pass
                                
                                # Translate (async)
                                asyncio.create_task(
                                    process_and_translate(final_text)
                                )
                            else:
                                # Interim result
                                try:
                                    await websocket.send_json({
                                        "type": "partial",
                                        "text": current_sentence + transcript,
                                    })
                                except Exception:
                                    pass
                    
                    elif msg_type == "PartialTranscript":
                        # Partial/interim results (v2 compatibility)
                        transcript = data.get("text", "")
                        if transcript:
                            try:
                                await websocket.send_json({
                                    "type": "partial",
                                    "text": current_sentence + transcript,
                                })
                            except Exception:
                                pass
                    
                    elif msg_type == "FinalTranscript":
                        # Final transcript (v2 compatibility)
                        transcript = data.get("text", "")
                        if transcript:
                            current_sentence += transcript + " "
                            final_text = current_sentence.strip()
                            current_sentence = ""
                            
                            try:
                                await websocket.send_json({
                                    "type": "original_complete",
                                    "text": final_text,
                                })
                            except Exception:
                                pass
                            
                            asyncio.create_task(
                                process_and_translate(final_text)
                            )
                    
                    elif msg_type == "Termination":
                        logger.info(f"AssemblyAI session terminated: {data.get('audio_duration_seconds')}s")
                        break
                    
                    elif msg_type == "Error":
                        error_msg = data.get("error", "Unknown error")
                        logger.error(f"AssemblyAI error: {error_msg}")
                        try:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"AssemblyAI: {error_msg}",
                            })
                        except Exception:
                            pass
                        
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            if not ws_closed:
                logger.error(f"AssemblyAI listener error: {e}")

    try:
        # Wait for config message first
        config_received = False
        while not config_received:
            message = await websocket.receive()
            if message["type"] == "websocket.receive" and "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == "config":
                    target_lang = data.get("targetLang", "en")
                    source_lang = data.get("sourceLang", "en")
                    translation_provider = data.get("translationProvider", "deepl")
                    quality_mode = data.get("qualityMode", "fast")
                    sample_rate = data.get("sampleRate", 16000)
                    config_received = True
                    logger.info(f"AssemblyAI config: {source_lang} -> {target_lang}, provider: {translation_provider}, quality: {quality_mode}")
        
        # Validate language support (including "auto")
        if source_lang not in SUPPORTED_LANGUAGES:
            await websocket.send_json({
                "type": "error",
                "message": f"AssemblyAI streaming does not support '{source_lang}'. Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}"
            })
            await websocket.close()
            return
        
        # Build AssemblyAI URL with proper v3 parameters
        url = build_assemblyai_url(source_lang, sample_rate, quality_mode)
        
        logger.info(f"Connecting to AssemblyAI: {url}")
        
        # Connect with authorization header
        assemblyai_ws = await websockets.connect(
            url,
            additional_headers={"Authorization": assemblyai_api_key},
            ping_interval=20,
            ping_timeout=10,
        )
        
        logger.info("Connected to AssemblyAI!")
        
        # Send ready message
        await websocket.send_json({
            "type": "ready",
            "message": "AssemblyAI transcription ready",
        })
        
        # Start listener task
        listen_task = asyncio.create_task(listen_to_assemblyai())

        # Handle client messages
        while True:
            try:
                message = await websocket.receive()
                
                if message["type"] == "websocket.disconnect":
                    break
                
                if message["type"] == "websocket.receive":
                    if "text" in message:
                        data = json.loads(message["text"])
                        msg_type = data.get("type", "")
                        
                        if msg_type == "audio":
                            # Base64 encoded PCM16 audio
                            audio_b64 = data.get("data", "")
                            if audio_b64 and assemblyai_ws:
                                audio_bytes = base64.b64decode(audio_b64)
                                await assemblyai_ws.send(audio_bytes)
                        
                        elif msg_type == "stop":
                            # Send terminate message to AssemblyAI
                            if assemblyai_ws:
                                try:
                                    await assemblyai_ws.send(json.dumps({"type": "Terminate"}))
                                except Exception:
                                    pass
                            break
                    
                    elif "bytes" in message:
                        # Raw binary audio
                        if assemblyai_ws:
                            await assemblyai_ws.send(message["bytes"])

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                pass
            except Exception as e:
                logger.error(f"Error handling client message: {e}")
                break

    except websockets.exceptions.InvalidStatusCode as e:
        logger.error(f"AssemblyAI rejected connection: {e.status_code}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"AssemblyAI connection failed (HTTP {e.status_code}). Check API key.",
            })
        except Exception:
            pass
    except Exception as e:
        logger.error(f"AssemblyAI error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })
        except Exception:
            pass

    finally:
        ws_closed = True
        if listen_task:
            listen_task.cancel()
            try:
                await listen_task
            except asyncio.CancelledError:
                pass
        if assemblyai_ws:
            try:
                await assemblyai_ws.close()
            except Exception:
                pass
        logger.info("AssemblyAI session ended")
