"""
WebSocket endpoint for real-time transcription using Google Cloud Speech-to-Text.
Google provides streaming recognition with interim results.
"""

import asyncio
import json
import logging
import os

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from google.cloud import speech_v1
from google.api_core.gapic_v1 import client_info as grpc_client_info

from ..config import settings
from ..services.translation_service import translation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["google"])


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


@router.websocket("/google-speech")
async def google_transcription(websocket: WebSocket) -> None:
    """
    Real-time transcription using Google Cloud Speech-to-Text.
    
    Supports interim results for near-real-time streaming.
    """
    await websocket.accept()
    logger.info("Client connected to Google Speech-to-Text endpoint")

    # Get credentials path at runtime
    google_credentials = os.getenv("GOOGLE_CLOUD_CREDENTIALS", "")
    
    if not google_credentials:
        await websocket.send_json({
            "type": "error",
            "message": "GOOGLE_CLOUD_CREDENTIALS not configured. Set up Google Cloud credentials."
        })
        await websocket.close()
        return

    target_lang = "en"
    current_sentence = ""
    last_final = ""
    translation_provider = "deepl"  # Default to DeepL

    try:
        # Initialize Google Speech-to-Text client
        try:
            client = speech_v1.SpeechClient()
        except Exception as e:
            logger.error(f"Failed to initialize Google Speech client: {e}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Failed to initialize Google: {str(e)}"
                })
            except:
                pass
            return
        
        try:
            config = speech_v1.RecognitionConfig(
                encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=24000,
                language_code="en-US",  # Will update based on config
                enable_automatic_punctuation=True,
                model="latest_short",  # FASTER: Optimized for short utterances
                use_enhanced=True,  # Enhanced model for better quality
            )
        except Exception as e:
            logger.error(f"Failed to create Google config: {e}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Failed to configure Google: {str(e)}"
                })
            except:
                pass
            return

        streaming_config = speech_v1.StreamingRecognitionConfig(
            config=config,
            interim_results=True,  # KEY: Get interim results!
        )

        async def process_audio_stream():
            """Process audio stream from client and send to Google."""
            nonlocal target_lang, current_sentence, last_final
            
            try:
                requests = []
                
                # First request with config
                requests.append(
                    speech_v1.StreamingRecognizeRequest(streaming_config=streaming_config)
                )
                
                while True:
                    try:
                        message = await websocket.receive()
                        
                        if message["type"] == "websocket.receive":
                            if "text" in message:
                                # JSON message (config)
                                data = json.loads(message["text"])
                                msg_type = data.get("type", "")
                                
                                if msg_type == "config":
                                    target_lang = data.get("targetLang", "en")
                                    source_lang = data.get("sourceLang", "en")
                                    translation_provider = data.get("translationProvider", "deepl")
                                    logger.info(f"Translation provider: {translation_provider}")
                                    
                                    # Update language code
                                    lang_map = {
                                        "en": "en-US",
                                        "el": "el-GR",
                                        "fr": "fr-FR",
                                        "de": "de-DE",
                                        "es": "es-ES",
                                        "it": "it-IT",
                                        "pt": "pt-BR",
                                        "ja": "ja-JP",
                                        "zh": "zh-CN",
                                    }
                                    
                                    lang_code = lang_map.get(source_lang, "en-US")
                                    config.language_code = lang_code
                                    logger.info(f"Language set to: {lang_code}")
                                
                                elif msg_type == "stop":
                                    break
                            
                            elif "bytes" in message:
                                # Binary audio data
                                audio_content = message["bytes"]
                                request = speech_v1.StreamingRecognizeRequest(
                                    audio_content=audio_content
                                )
                                requests.append(request)
                    
                    except WebSocketDisconnect:
                        break
                    except Exception as e:
                        logger.error(f"Stream error: {e}")
                        break
                
                # Process requests to Google
                responses = client.streaming_recognize(requests)
                
                for response in responses:
                    if not response.results:
                        continue
                    
                    result = response.results[0]
                    transcript = result.alternatives[0].transcript if result.alternatives else ""
                    
                    if result.is_final:
                        # Final result
                        if transcript and transcript != last_final:
                            current_sentence += transcript + " "
                            last_final = transcript
                            
                            await websocket.send_json({
                                "type": "partial",
                                "text": current_sentence.strip(),
                            })
                    else:
                        # Interim result - show immediately!
                        await websocket.send_json({
                            "type": "partial",
                            "text": (current_sentence + transcript).strip(),
                        })
                
                # Send complete
                if current_sentence.strip():
                    final_text = current_sentence.strip()
                    current_sentence = ""
                    
                    await websocket.send_json({
                        "type": "original_complete",
                        "text": final_text,
                    })
                    
                    # Process in background
                    asyncio.create_task(
                        process_and_translate(websocket, final_text, target_lang, translation_provider)
                    )
            
            except Exception as e:
                logger.error(f"Google API error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })

        async def process_and_translate(ws: WebSocket, text: str, target: str, provider: str = "deepl") -> None:
            """Check syntax and translate (runs in background)."""
            try:
                # Step 1: Syntax check
                corrected = await check_and_fix_syntax(text)
                
                if corrected != text:
                    await ws.send_json({
                        "type": "original_corrected",
                        "original": text,
                        "corrected": corrected,
                    })
                    text = corrected
                
                # Step 2: Translate (uses DeepL if available, else OpenAI)
                translation = await translation_service.translate_text(
                    text=text,
                    target_lang=target,
                    source_lang=None,
                                    provider=provider,
                )
                
                await ws.send_json({
                    "type": "translation",
                    "original": text,
                    "translation": translation,
                })
                
            except Exception as e:
                logger.error(f"Process/translate error: {e}")

        # Start processing
        await process_audio_stream()

    except Exception as e:
        logger.error(f"Google Speech-to-Text error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e),
        })

    finally:
        logger.info("Google Speech connection closed")
