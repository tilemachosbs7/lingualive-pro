"""
WebSocket endpoint for real-time transcription using Deepgram.
Optimized for Deepgram Nova + DeepL combo with:
- Two-pass translation (fast on is_final, refine on speech_final)
- Latest-wins cancellation (drop old translations)
- Session-level timing metrics
- Feature-flagged syntax correction
- Configurable endpointing
"""

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from typing import Optional

import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import settings
from ..services.translation_service import translation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["deepgram"])

DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"


async def check_and_fix_syntax(text: str, timeout_ms: int = 3000) -> str:
    """Quick syntax/grammar check using GPT.
    
    Only called if ENABLE_SYNTAX_FIX=true and in quality mode.
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
        "max_tokens": 300,
    }
    
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json",
    }
    
    timeout_s = timeout_ms / 1000.0
    
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
    except asyncio.TimeoutError:
        logger.warning(f"Syntax fix timed out after {timeout_s}s")
        return text
    except Exception as e:
        logger.warning(f"Syntax check failed (continuing with original): {e}")
        return text


@router.websocket("/deepgram")
async def deepgram_transcription(websocket: WebSocket) -> None:
    """
    Real-time transcription using Deepgram Nova + DeepL.
    
    Features:
    - Two-pass translation: fast on is_final, refine on speech_final
    - Latest-wins task cancellation
    - Session timing metrics
    - Configurable endpointing (utterance_end_ms)
    """
    await websocket.accept()
    
    # Generate session ID for log correlation
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"[{session_id}] Client connected to Deepgram endpoint")

    # Get API key at runtime
    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY", "")
    
    if not deepgram_api_key:
        await websocket.send_json({
            "type": "error",
            "message": "DEEPGRAM_API_KEY not configured"
        })
        await websocket.close()
        return

    # Session state
    target_lang = "en"
    source_lang: Optional[str] = None  # Source language hint for DeepL
    translation_provider = "deepl"
    quality_mode = "fast"  # fast or quality
    current_sentence = ""
    deepgram_ws = None
    listen_task = None
    ws_closed = False
    
    # Latest-wins task management
    current_translate_task: Optional[asyncio.Task] = None
    translate_task_lock = asyncio.Lock()
    
    # Timing metrics
    session_start_time = time.perf_counter()
    total_translations = 0
    total_translation_time_ms = 0.0

    async def safe_send(data: dict) -> bool:
        """Send JSON to client with error handling."""
        nonlocal ws_closed
        if ws_closed:
            return False
        try:
            await websocket.send_json(data)
            return True
        except Exception as e:
            logger.debug(f"[{session_id}] Send failed: {e}")
            return False

    async def process_and_translate(
        text: str,
        is_refine_pass: bool = False,
    ) -> None:
        """Process text and translate with optional syntax fix.
        
        Args:
            text: Text to translate
            is_refine_pass: True for speech_final (full sentence), False for is_final (fast)
        """
        nonlocal ws_closed, total_translations, total_translation_time_ms
        if ws_closed:
            return
        
        translate_start = time.perf_counter()
        processed_text = text
        
        try:
            # Syntax fix only in quality mode AND if enabled AND this is refine pass
            if (
                is_refine_pass
                and quality_mode == "quality"
                and settings.enable_syntax_fix
            ):
                corrected = await check_and_fix_syntax(
                    text, 
                    timeout_ms=settings.syntax_fix_timeout_ms
                )
                
                if corrected != text:
                    await safe_send({
                        "type": "original_corrected",
                        "original": text,
                        "corrected": corrected,
                    })
                    processed_text = corrected
            
            # Check for cancellation before translation
            if asyncio.current_task().cancelled():
                return
            
            # Translate with source language hint
            translation = await translation_service.translate_text(
                text=processed_text,
                target_lang=target_lang,
                source_lang=source_lang if source_lang != "auto" else None,
                provider=translation_provider,
                timeout_ms=settings.translation_timeout_ms,
            )
            
            # Check for cancellation after translation
            if asyncio.current_task().cancelled():
                return
            
            translate_duration_ms = (time.perf_counter() - translate_start) * 1000
            total_translations += 1
            total_translation_time_ms += translate_duration_ms
            
            # Send translation with metadata
            await safe_send({
                "type": "translation",
                "original": processed_text,
                "translation": translation,
                "is_refine": is_refine_pass,  # Optional field for HUD
                "latency_ms": round(translate_duration_ms, 1),  # Optional timing
            })
            
            logger.debug(
                f"[{session_id}] Translation: {translate_duration_ms:.1f}ms "
                f"(refine={is_refine_pass}, provider={translation_provider})"
            )
            
        except asyncio.CancelledError:
            logger.debug(f"[{session_id}] Translation cancelled (newer request)")
            raise
        except TimeoutError as e:
            logger.warning(f"[{session_id}] Translation timeout: {e}")
            await safe_send({
                "type": "error",
                "message": f"Translation timeout ({settings.translation_timeout_ms}ms)",
            })
        except Exception as e:
            logger.error(f"[{session_id}] Process/translate error: {e}")
            await safe_send({
                "type": "error",
                "message": f"Translation failed: {str(e)[:100]}",
            })

    async def schedule_translation(text: str, is_refine: bool) -> None:
        """Schedule translation with latest-wins cancellation."""
        nonlocal current_translate_task
        
        async with translate_task_lock:
            # Cancel previous task if still running (latest wins)
            if current_translate_task and not current_translate_task.done():
                current_translate_task.cancel()
                try:
                    await current_translate_task
                except asyncio.CancelledError:
                    pass
            
            # Create new task
            current_translate_task = asyncio.create_task(
                process_and_translate(text, is_refine_pass=is_refine)
            )

    async def listen_to_deepgram() -> None:
        """Listen for Deepgram responses with two-pass translation."""
        nonlocal current_sentence, ws_closed
        
        accumulated_text = ""  # Text since last speech_final
        
        try:
            async for message in deepgram_ws:
                if ws_closed:
                    break
                try:
                    data = json.loads(message)
                    
                    if data.get("type") == "Results":
                        channel = data.get("channel", {})
                        alternatives = channel.get("alternatives", [])
                        
                        if alternatives:
                            transcript = alternatives[0].get("transcript", "")
                            is_final = data.get("is_final", False)
                            speech_final = data.get("speech_final", False)
                            
                            if transcript:
                                if is_final:
                                    # Accumulate finalized text
                                    accumulated_text += transcript + " "
                                    current_sentence = accumulated_text.strip()
                                    
                                    # Send partial update
                                    await safe_send({
                                        "type": "partial",
                                        "text": current_sentence,
                                    })
                                    
                                    # === TWO-PASS TRANSLATION ===
                                    if settings.enable_two_pass_translation:
                                        # FAST PASS: Translate immediately on is_final
                                        # This gives user quick feedback
                                        if not speech_final and len(current_sentence) > 5:
                                            await schedule_translation(
                                                current_sentence, 
                                                is_refine=False
                                            )
                                    
                                    if speech_final:
                                        # End of utterance - send complete original
                                        final_text = accumulated_text.strip()
                                        accumulated_text = ""
                                        
                                        await safe_send({
                                            "type": "original_complete",
                                            "text": final_text,
                                        })
                                        
                                        # REFINE PASS: Full sentence translation
                                        await schedule_translation(
                                            final_text, 
                                            is_refine=True
                                        )
                                else:
                                    # Interim result - show preview
                                    preview = accumulated_text + transcript
                                    await safe_send({
                                        "type": "partial",
                                        "text": preview.strip(),
                                    })
                                        
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            if not ws_closed:
                logger.error(f"[{session_id}] Deepgram listener error: {e}")

    try:
        # Build Deepgram URL with optimized parameters
        utterance_end_ms = settings.deepgram_utterance_end_ms
        vad_events = str(settings.deepgram_vad_events).lower()
        
        params = [
            "model=nova-2",
            "encoding=linear16",
            "sample_rate=24000",
            "channels=1",
            "punctuate=true",
            "interim_results=true",
            f"utterance_end_ms={utterance_end_ms}",  # Faster endpointing
            f"vad_events={vad_events}",  # Voice activity detection
            "smart_format=true",  # Better punctuation
        ]
        url = f"{DEEPGRAM_URL}?{'&'.join(params)}"
        
        logger.info(f"[{session_id}] Connecting to Deepgram (utterance_end_ms={utterance_end_ms})...")
        
        # Connect with authorization header
        deepgram_ws = await websockets.connect(
            url,
            additional_headers={"Authorization": f"Token {deepgram_api_key}"},
            ping_interval=20,
            ping_timeout=10,
        )
        
        logger.info(f"[{session_id}] Connected to Deepgram!")
        
        # Send ready message
        await safe_send({
            "type": "ready",
            "message": "Deepgram transcription ready",
            "session_id": session_id,  # Optional for client-side correlation
        })
        
        # Start listener task
        listen_task = asyncio.create_task(listen_to_deepgram())

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
                        
                        if msg_type == "config":
                            target_lang = data.get("targetLang", "en")
                            source_lang = data.get("sourceLang", None)  # Get source lang!
                            translation_provider = data.get("translationProvider", "deepl")
                            quality_mode = data.get("qualityMode", "fast")
                            logger.info(
                                f"[{session_id}] Config: {source_lang} -> {target_lang}, "
                                f"provider={translation_provider}, quality={quality_mode}"
                            )
                        
                        elif msg_type == "audio":
                            audio_b64 = data.get("data", "")
                            if audio_b64 and deepgram_ws:
                                audio_bytes = base64.b64decode(audio_b64)
                                await deepgram_ws.send(audio_bytes)
                        
                        elif msg_type == "stop":
                            break
                    
                    elif "bytes" in message:
                        if deepgram_ws:
                            await deepgram_ws.send(message["bytes"])

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                pass
            except Exception as e:
                logger.error(f"[{session_id}] Error: {e}")
                break

    except websockets.exceptions.InvalidStatusCode as e:
        logger.error(f"[{session_id}] Deepgram rejected connection: {e.status_code}")
        await safe_send({
            "type": "error",
            "message": f"Deepgram connection failed (HTTP {e.status_code}). Check API key.",
        })
    except Exception as e:
        logger.error(f"[{session_id}] Deepgram error: {e}")
        await safe_send({
            "type": "error",
            "message": str(e),
        })

    finally:
        ws_closed = True
        
        # Cancel any pending translation
        if current_translate_task and not current_translate_task.done():
            current_translate_task.cancel()
        
        if listen_task:
            listen_task.cancel()
            try:
                await listen_task
            except asyncio.CancelledError:
                pass
        
        if deepgram_ws:
            try:
                await deepgram_ws.close()
            except Exception:
                pass
        
        # Log session metrics
        session_duration = time.perf_counter() - session_start_time
        avg_translation = (
            total_translation_time_ms / total_translations 
            if total_translations > 0 else 0
        )
        logger.info(
            f"[{session_id}] Session ended: {session_duration:.1f}s, "
            f"{total_translations} translations, avg {avg_translation:.1f}ms/translation"
        )
