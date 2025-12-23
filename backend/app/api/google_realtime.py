"""
WebSocket endpoint for real-time transcription using Google Cloud Speech-to-Text.
Uses streaming recognition with interim results.
"""

import asyncio
import base64
import json
import logging
import os
import queue
import threading

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from google.cloud import speech

from ..config import settings
from ..services.translation_service import translation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["google"])


async def check_and_fix_syntax(text: str) -> str:
    """Quick syntax/grammar check using GPT."""
    # Skip if OpenAI key is not available
    openai_key = settings.openai_api_key
    if not openai_key:
        logger.debug("Skipping syntax fix - OpenAI API key not configured")
        return text
    
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
        logger.warning(f"Syntax fix failed (continuing with original): {e}")
        return text  # Return original text if syntax fix fails


@router.websocket("/google-speech")
async def google_transcription(websocket: WebSocket) -> None:
    """
    Real-time transcription using Google Cloud Speech-to-Text.
    """
    await websocket.accept()
    logger.info("Client connected to Google Speech-to-Text endpoint")

    # Get credentials path
    google_credentials = os.getenv("GOOGLE_CLOUD_CREDENTIALS", "")
    
    if not google_credentials:
        await websocket.send_json({
            "type": "error",
            "message": "GOOGLE_CLOUD_CREDENTIALS not configured."
        })
        await websocket.close()
        return

    # Set credentials environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials

    target_lang = "en"
    source_lang = "en"
    translation_provider = "deepl"
    sample_rate = 16000  # Default, will be updated from config message
    current_sentence = ""
    
    # Thread-safe queues for communication
    audio_queue = queue.Queue()
    result_queue = queue.Queue()
    stop_event = threading.Event()

    def audio_generator():
        """Generate audio chunks for Google Speech API."""
        while not stop_event.is_set():
            try:
                # Get audio with timeout to check stop event
                audio_data = audio_queue.get(timeout=0.1)
                if audio_data is None:  # Stop signal
                    break
                yield speech.StreamingRecognizeRequest(audio_content=audio_data)
            except queue.Empty:
                continue

    def run_speech_recognition(language_code: str, sample_rate_hz: int):
        """Run Google Speech recognition in a separate thread."""
        try:
            client = speech.SpeechClient()
            
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate_hz,
                language_code=language_code,
                enable_automatic_punctuation=True,
                model="latest_short",
            )
            
            streaming_config = speech.StreamingRecognitionConfig(
                config=config,
                interim_results=True,
            )

            # Call streaming_recognize with config and audio generator
            responses = client.streaming_recognize(
                config=streaming_config,
                requests=audio_generator()
            )
            
            for response in responses:
                if stop_event.is_set():
                    break
                    
                for result in response.results:
                    if not result.alternatives:
                        continue
                    
                    transcript = result.alternatives[0].transcript
                    is_final = result.is_final
                    
                    result_queue.put({
                        "transcript": transcript,
                        "is_final": is_final
                    })
                    
        except Exception as e:
            logger.error(f"Google Speech error: {e}")
            result_queue.put({"error": str(e)})

    async def process_and_translate(text: str, target: str, provider: str, quality_mode: str = "fast", source: str = None) -> None:
        """Check syntax and translate."""
        try:
            # Syntax check only in quality mode (skip in fast mode for lower latency)
            corrected = text
            if quality_mode == "quality":
                corrected = await check_and_fix_syntax(text)
            
            if corrected != text:
                await websocket.send_json({
                    "type": "original_corrected",
                    "original": text,
                    "corrected": corrected,
                })
                text = corrected
            
            # Translate (pass source_lang to skip auto-detect and reduce latency)
            translation = await translation_service.translate_text(
                text=text,
                target_lang=target,
                source_lang=source if source and source != "auto" else None,
                provider=provider,
            )
            
            await websocket.send_json({
                "type": "translation",
                "original": text,
                "translation": translation,
            })
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            # Send error to client for better debugging
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Translation failed: {str(e)}"
                })
            except Exception:
                pass  # Client may have disconnected

    try:
        # Language code mapping
        lang_map = {
            "auto": "en-US", "en": "en-US", "el": "el-GR", "fr": "fr-FR",
            "de": "de-DE", "es": "es-ES", "it": "it-IT", "pt": "pt-BR",
            "ja": "ja-JP", "zh": "zh-CN", "ko": "ko-KR", "ru": "ru-RU",
            "ar": "ar-SA", "hi": "hi-IN", "nl": "nl-NL", "pl": "pl-PL",
            "tr": "tr-TR", "sv": "sv-SE", "da": "da-DK", "fi": "fi-FI",
            "no": "nb-NO", "cs": "cs-CZ", "hu": "hu-HU", "ro": "ro-RO",
            "sk": "sk-SK", "uk": "uk-UA", "bg": "bg-BG", "hr": "hr-HR",
            "sl": "sl-SI", "et": "et-EE", "lv": "lv-LV", "lt": "lt-LT",
            "vi": "vi-VN", "th": "th-TH", "id": "id-ID", "ms": "ms-MY",
            "he": "iw-IL", "fa": "fa-IR", "ur": "ur-PK", "bn": "bn-BD",
            "ta": "ta-IN", "te": "te-IN", "mr": "mr-IN", "gu": "gu-IN",
        }
        
        # Start speech recognition thread
        speech_thread = None
        
        # Main loop
        while True:
            # Check for results from speech thread
            try:
                while True:
                    result = result_queue.get_nowait()
                    
                    if "error" in result:
                        await websocket.send_json({
                            "type": "error",
                            "message": result["error"]
                        })
                        continue
                    
                    transcript = result["transcript"]
                    is_final = result["is_final"]
                    
                    if is_final:
                        # Final result
                        if transcript:
                            current_sentence += transcript + " "
                            final_text = current_sentence.strip()
                            current_sentence = ""
                            
                            await websocket.send_json({
                                "type": "original_complete",
                                "text": final_text,
                            })
                            
                            # Translate in background
                            asyncio.create_task(
                                process_and_translate(final_text, target_lang, translation_provider, quality_mode, source_lang)
                            )
                    else:
                        # Interim result
                        await websocket.send_json({
                            "type": "partial",
                            "text": (current_sentence + transcript).strip(),
                        })
                        
            except queue.Empty:
                pass
            
            # Receive websocket message with timeout
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=0.05
                )
                
                if message["type"] == "websocket.receive":
                    if "text" in message:
                        data = json.loads(message["text"])
                        msg_type = data.get("type", "")
                        
                        if msg_type == "config":
                            target_lang = data.get("targetLang", "en")
                            source_lang = data.get("sourceLang", "en")
                            translation_provider = data.get("translationProvider", "deepl")
                            quality_mode = data.get("qualityMode", "fast")  # fast or quality
                            # Get sample rate from config message (default to 16000 if not provided)
                            config_sample_rate = data.get("sampleRate", 16000)
                            if isinstance(config_sample_rate, (int, float)) and config_sample_rate > 0:
                                sample_rate = int(config_sample_rate)
                            logger.info(f"Google STT: {source_lang} ({lang_map.get(source_lang, 'en-US')}) -> {target_lang}, quality={quality_mode}, sample_rate={sample_rate}")
                            
                            language_code = lang_map.get(source_lang, "en-US")
                            
                            # Start speech thread if not running
                            if speech_thread is None or not speech_thread.is_alive():
                                speech_thread = threading.Thread(
                                    target=run_speech_recognition,
                                    args=(language_code, sample_rate),
                                    daemon=True
                                )
                                speech_thread.start()
                        
                        elif msg_type == "audio":
                            # Base64 encoded audio
                            audio_b64 = data.get("data", "")
                            if audio_b64:
                                audio_bytes = base64.b64decode(audio_b64)
                                audio_queue.put(audio_bytes)
                        
                        elif msg_type == "stop":
                            break
                    
                    elif "bytes" in message:
                        # Raw binary audio
                        audio_queue.put(message["bytes"])
                        
                elif message["type"] == "websocket.disconnect":
                    break
                    
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                break
                
    except Exception as e:
        logger.error(f"Google Speech endpoint error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except Exception:
            pass
            
    finally:
        # Cleanup
        stop_event.set()
        audio_queue.put(None)  # Stop signal
        logger.info("Google Speech connection closed")
