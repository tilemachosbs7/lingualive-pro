"""
WebSocket endpoint for real-time audio transcription and translation.
"""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import settings
from ..services.translation_service import translation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["realtime"])

# OpenAI Realtime API configuration
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"


async def check_and_fix_syntax(text: str) -> str:
    """Quick syntax/grammar check using GPT."""
    import httpx
    
    prompt = "Fix any transcription errors or grammar issues. Only output the corrected text, nothing else. If text is correct, output it unchanged."
    
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
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


@router.websocket("/realtime")
async def realtime_transcription(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time audio transcription and translation.
    
    Client sends:
    - {"type": "config", "sourceLang": "auto", "targetLang": "el"}
    - {"type": "audio", "data": "<base64 PCM16 audio>"}
    - {"type": "stop"}
    
    Server sends:
    - {"type": "partial", "text": "..."} - Streaming partial transcript (INSTANT)
    - {"type": "transcript", "original": "...", "translation": "..."} - Final with translation
    - {"type": "error", "message": "..."}
    """
    await websocket.accept()
    logger.info("Client connected to realtime endpoint")

    source_lang = "auto"
    target_lang = "en"
    openai_ws = None
    listen_task = None
    current_partial = ""  # Buffer for streaming text
    translation_provider = "deepl"  # Default to DeepL

    try:
        import websockets

        # Connect to OpenAI Realtime API
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        try:
            openai_ws = await websockets.connect(
                OPENAI_REALTIME_URL,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
            )
            logger.info("Connected to OpenAI Realtime API")
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI: {e}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Failed to connect to OpenAI: {str(e)}"
                })
            except Exception:
                pass
            return

        # Configure session - Most aggressive VAD settings
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.15,
                    "prefix_padding_ms": 100,
                    "silence_duration_ms": 200,
                },
            },
        }
        await openai_ws.send(json.dumps(session_config))

        async def listen_to_openai() -> None:
            """Listen for responses from OpenAI and forward to client."""
            nonlocal target_lang, current_partial
            try:
                async for message in openai_ws:
                    try:
                        data = json.loads(message)
                        event_type = data.get("type", "")

                        # STREAMING: Partial transcription (word by word)
                        if event_type == "conversation.item.input_audio_transcription.delta":
                            delta = data.get("delta", "")
                            if delta:
                                current_partial += delta
                                # Send INSTANT partial update
                                await websocket.send_json({
                                    "type": "partial",
                                    "text": current_partial,
                                })

                        # COMPLETE: Full transcription ready
                        elif event_type == "conversation.item.input_audio_transcription.completed":
                            transcript = data.get("transcript", "").strip()
                            if not transcript:
                                transcript = current_partial.strip()
                            
                            current_partial = ""  # Reset buffer
                            
                            if transcript:
                                # Send the raw transcript immediately
                                await websocket.send_json({
                                    "type": "original_complete",
                                    "text": transcript,
                                })
                                
                                # Now do syntax check + translation in background
                                asyncio.create_task(
                                    process_and_translate(websocket, transcript, target_lang, translation_provider)
                                )

                        elif event_type == "error":
                            error = data.get("error", {})
                            logger.error(f"OpenAI Realtime error: {error}")
                            await websocket.send_json({
                                "type": "error",
                                "message": error.get("message", "Unknown error"),
                            })

                    except json.JSONDecodeError:
                        pass

            except Exception as e:
                logger.error(f"OpenAI listener error: {e}")

        async def process_and_translate(ws: WebSocket, text: str, target: str, provider: str = "deepl") -> None:
            """Check syntax and translate (runs in background)."""
            try:
                # Step 1: Syntax check (quick)
                corrected = await check_and_fix_syntax(text)
                
                # If corrected is different, send update
                if corrected != text:
                    await ws.send_json({
                        "type": "original_corrected",
                        "original": text,
                        "corrected": corrected,
                    })
                    text = corrected
                
                # Step 2: Translate
                translation = await translation_service.translate_text(
                    text=text,
                    target_lang=target,
                    source_lang=None,
                                    provider=provider,
                )
                
                # Send final translation
                await ws.send_json({
                    "type": "translation",
                    "original": text,
                    "translation": translation,
                })
                
            except Exception as e:
                logger.error(f"Process/translate error: {e}")
                await ws.send_json({
                    "type": "translation",
                    "original": text,
                    "translation": f"[Error: {e}]",
                })

        # Start listening to OpenAI in background
        listen_task = asyncio.create_task(listen_to_openai())

        # Handle messages from client
        while True:
            try:
                message = await websocket.receive_json()
                msg_type = message.get("type", "")

                if msg_type == "config":
                    source_lang = message.get("sourceLang", "auto")
                    target_lang = message.get("targetLang", "en")
                    logger.info(f"Configured: {source_lang} -> {target_lang}")
                    translation_provider = message.get("translationProvider", "deepl")
                    logger.info(f"Translation provider: {translation_provider}")

                elif msg_type == "audio":
                    # Forward audio to OpenAI
                    audio_b64 = message.get("data", "")
                    if audio_b64:
                        await openai_ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": audio_b64,
                        }))

                elif msg_type == "stop":
                    break

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                logger.debug("Failed to parse JSON from client")
            except Exception as e:
                logger.error(f"Error handling client message: {e}")
                try:
                    await websocket.send_json({"type": "error", "message": str(e)})
                except Exception:
                    pass
                break

    except Exception as e:
        logger.error(f"Realtime WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })
        except Exception:
            pass

    finally:
        # Cleanup
        if listen_task:
            listen_task.cancel()
            try:
                await listen_task
            except asyncio.CancelledError:
                pass

        if openai_ws:
            await openai_ws.close()

        logger.info("Realtime connection closed")
