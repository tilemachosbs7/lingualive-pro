"""
OpenAI Realtime API service for streaming audio transcription.
Uses WebSocket for low-latency real-time processing.
"""

import asyncio
import base64
import json
import logging
from typing import AsyncGenerator, Callable

import websockets
from websockets.client import WebSocketClientProtocol

from ..config import settings

logger = logging.getLogger(__name__)

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"


class RealtimeSession:
    """Manages a real-time audio transcription session with OpenAI."""

    def __init__(
        self,
        source_lang: str = "auto",
        target_lang: str = "en",
        on_transcript: Callable[[str, str], None] | None = None,
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.on_transcript = on_transcript
        self.ws: WebSocketClientProtocol | None = None
        self._running = False
        self._transcript_buffer = ""
        self._tasks: list[asyncio.Task] = []

    async def connect(self) -> None:
        """Establish WebSocket connection to OpenAI Realtime API."""
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        try:
            self.ws = await websockets.connect(
                OPENAI_REALTIME_URL,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
            )
            self._running = True
            logger.info("Connected to OpenAI Realtime API")

            # Configure the session for audio transcription
            await self._configure_session()

            # Start listening for responses
            self._tasks.append(asyncio.create_task(self._listen_responses()))

        except Exception as e:
            logger.error(f"Failed to connect to OpenAI Realtime API: {e}")
            raise

    async def _configure_session(self) -> None:
        """Configure the realtime session for transcription."""
        if not self.ws:
            return

        # Session configuration
        config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": f"You are a real-time transcription and translation assistant. Listen to the audio and transcribe it accurately. Then translate the transcription to {self.target_lang}. Respond with the format: [ORIGINAL] <transcription> [TRANSLATION] <translation>",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
                "temperature": 0.6,
            },
        }

        await self.ws.send(json.dumps(config))
        logger.info("Session configured for transcription")

    async def send_audio(self, audio_data: bytes) -> None:
        """Send audio data to the realtime API."""
        if not self.ws or not self._running:
            return

        # Audio must be base64 encoded
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        message = {
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
        }

        try:
            await self.ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send audio: {e}")

    async def commit_audio(self) -> None:
        """Commit the audio buffer to trigger processing."""
        if not self.ws or not self._running:
            return

        message = {"type": "input_audio_buffer.commit"}
        try:
            await self.ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to commit audio: {e}")

    async def _listen_responses(self) -> None:
        """Listen for responses from the realtime API."""
        if not self.ws:
            return

        try:
            async for message in self.ws:
                if not self._running:
                    break

                try:
                    data = json.loads(message)
                    await self._handle_event(data)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse message: {message}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in response listener: {e}")
        finally:
            self._running = False

    async def _handle_event(self, event: dict) -> None:
        """Handle events from the realtime API."""
        event_type = event.get("type", "")

        if event_type == "session.created":
            logger.info("Realtime session created")

        elif event_type == "session.updated":
            logger.info("Realtime session updated")

        elif event_type == "input_audio_buffer.speech_started":
            logger.debug("Speech started")

        elif event_type == "input_audio_buffer.speech_stopped":
            logger.debug("Speech stopped")

        elif event_type == "conversation.item.input_audio_transcription.completed":
            # This is the transcription result
            transcript = event.get("transcript", "")
            if transcript and self.on_transcript:
                self.on_transcript(transcript, "")
                logger.info(f"Transcription: {transcript}")

        elif event_type == "response.audio_transcript.delta":
            # Streaming text response
            delta = event.get("delta", "")
            self._transcript_buffer += delta

        elif event_type == "response.audio_transcript.done":
            # Complete response
            transcript = event.get("transcript", self._transcript_buffer)
            if transcript and self.on_transcript:
                self.on_transcript("", transcript)
            self._transcript_buffer = ""

        elif event_type == "response.text.delta":
            # Text response delta
            delta = event.get("delta", "")
            self._transcript_buffer += delta

        elif event_type == "response.text.done":
            # Parse the response for original and translation
            text = event.get("text", self._transcript_buffer)
            if text:
                self._parse_and_emit(text)
            self._transcript_buffer = ""

        elif event_type == "response.done":
            # Response complete
            if self._transcript_buffer:
                self._parse_and_emit(self._transcript_buffer)
                self._transcript_buffer = ""

        elif event_type == "error":
            error = event.get("error", {})
            logger.error(f"Realtime API error: {error}")

    def _parse_and_emit(self, text: str) -> None:
        """Parse response and emit transcript/translation."""
        if not self.on_transcript:
            return

        original = ""
        translation = ""

        if "[ORIGINAL]" in text and "[TRANSLATION]" in text:
            parts = text.split("[TRANSLATION]")
            original = parts[0].replace("[ORIGINAL]", "").strip()
            translation = parts[1].strip() if len(parts) > 1 else ""
        else:
            # Treat entire text as translation
            translation = text.strip()

        if original or translation:
            self.on_transcript(original, translation)

    async def close(self) -> None:
        """Close the realtime session."""
        self._running = False

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        if self.ws:
            await self.ws.close()
            self.ws = None

        logger.info("Realtime session closed")
