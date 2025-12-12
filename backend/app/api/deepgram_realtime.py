"""
WebSocket endpoint for real-time transcription using Deepgram.
Uses raw websockets for reliable streaming connection.
"""

import asyncio
import base64
import json
import logging
import os

import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import settings
from ..services.translation_service import translation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["deepgram"])

DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"


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


@router.websocket("/deepgram")
async def deepgram_transcription(websocket: WebSocket) -> None:
    """
    Real-time transcription using Deepgram via raw websockets.
    """
    await websocket.accept()
    logger.info("Client connected to Deepgram endpoint")

    # Get API key at runtime
    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY", "")
    
    if not deepgram_api_key:
        await websocket.send_json({
            "type": "error",
            "message": "DEEPGRAM_API_KEY not configured"
        })
        await websocket.close()
        return

    target_lang = "en"
    translation_provider = "deepl"
    current_sentence = ""
    deepgram_ws = None
    listen_task = None
    ws_closed = False

    async def process_and_translate(text: str) -> None:
        """Check syntax and translate."""
        nonlocal ws_closed
        if ws_closed:
            return
            
        try:
            corrected = await check_and_fix_syntax(text)
            
            if corrected != text and not ws_closed:
                try:
                    await websocket.send_json({
                        "type": "original_corrected",
                        "original": text,
                        "corrected": corrected,
                    })
                except:
                    pass
                text = corrected
            
            translation = await translation_service.translate_text(
                text=text,
                target_lang=target_lang,
                source_lang=None,
                provider=translation_provider,
            )
            
            if not ws_closed:
                try:
                    await websocket.send_json({
                        "type": "translation",
                        "original": text,
                        "translation": translation,
                    })
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Process/translate error: {e}")

    async def listen_to_deepgram() -> None:
        """Listen for Deepgram responses."""
        nonlocal current_sentence, ws_closed
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
                                    current_sentence += transcript + " "
                                    
                                    try:
                                        await websocket.send_json({
                                            "type": "partial",
                                            "text": current_sentence.strip(),
                                        })
                                    except:
                                        pass
                                    
                                    if speech_final:
                                        final_text = current_sentence.strip()
                                        current_sentence = ""
                                        
                                        try:
                                            await websocket.send_json({
                                                "type": "original_complete",
                                                "text": final_text,
                                            })
                                        except:
                                            pass
                                        
                                        asyncio.create_task(
                                            process_and_translate(final_text)
                                        )
                                else:
                                    try:
                                        await websocket.send_json({
                                            "type": "partial",
                                            "text": current_sentence + transcript,
                                        })
                                    except:
                                        pass
                                        
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            if not ws_closed:
                logger.error(f"Deepgram listener error: {e}")

    try:
        # Build Deepgram URL with parameters
        params = [
            "model=nova-2",
            "encoding=linear16",
            "sample_rate=24000",
            "channels=1",
            "punctuate=true",
            "interim_results=true",
        ]
        url = f"{DEEPGRAM_URL}?{'&'.join(params)}"
        
        logger.info(f"Connecting to Deepgram...")
        
        # Connect with authorization header
        deepgram_ws = await websockets.connect(
            url,
            additional_headers={"Authorization": f"Token {deepgram_api_key}"},
            ping_interval=20,
            ping_timeout=10,
        )
        
        logger.info("Connected to Deepgram!")
        
        # Send ready message
        await websocket.send_json({
            "type": "ready",
            "message": "Deepgram transcription ready",
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
                            translation_provider = data.get("translationProvider", "deepl")
                            logger.info(f"Config: target={target_lang}, provider={translation_provider}")
                        
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
                logger.error(f"Error: {e}")
                break

    except websockets.exceptions.InvalidStatusCode as e:
        logger.error(f"Deepgram rejected connection: {e.status_code}")
        await websocket.send_json({
            "type": "error",
            "message": f"Deepgram connection failed (HTTP {e.status_code}). Check API key.",
        })
    except Exception as e:
        logger.error(f"Deepgram error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })
        except:
            pass

    finally:
        ws_closed = True
        if listen_task:
            listen_task.cancel()
        if deepgram_ws:
            try:
                await deepgram_ws.close()
            except:
                pass
        logger.info("Deepgram session ended")
