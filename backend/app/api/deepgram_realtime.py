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
from typing import Dict, List, Optional, Tuple

import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import settings
from ..services.translation_service import translation_service

logger = logging.getLogger(__name__)

# === DEBUG INSTRUMENTATION (toggle to enable verbose diagnostic logs) ===
DEBUG_STALLS = True  # Set False to disable debug logs

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
    
    # Log API key status (without exposing the key)
    logger.info(f"[{session_id}] DEEPGRAM_API_KEY present: {bool(deepgram_api_key)}")
    
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
    config_received = False  # AAA: Wait for config before Deepgram connection
    client_sample_rate = 16000  # AAA: Actual sample rate from client
    client_utterance_end_ms = 0  # AAA: Per-mode endpointing from client (0 = use default)
    playback_rate = 1.0  # AAA: Video playback speed (0.5x-2x)
    
    # Latest-wins task management - SEPARATE tasks for fast and refine!
    current_fast_task: Optional[asyncio.Task] = None  # Fast/partial translations (can be cancelled)
    current_refine_task: Optional[asyncio.Task] = None  # Refine translations (NEVER cancelled)
    translate_task_lock = asyncio.Lock()
    segment_lock = asyncio.Lock()  # Benefit: stable segment order without lost/duplicate segments
    send_lock = asyncio.Lock()  # Benefit: serialized WS sends prevent dropped/out-of-order messages
    pending_fast_task: Optional[asyncio.Task] = None
    last_translate_scheduled_at = 0.0
    
    # AAA: Background tasks (keepalive, heartbeat)
    keepalive_task: Optional[asyncio.Task] = None
    heartbeat_task: Optional[asyncio.Task] = None
    
    # AAA: Partial translation throttle
    last_partial_translate_time = 0.0
    
    # AAA: Backpressure state
    backpressure_engaged = False
    effective_min_partial_chars = settings.partial_translate_min_chars
    
    # AAA: Speed Mode (auto-enable when latency is high for x2 playback)
    # x2: Lower threshold (900ms) for earlier activation
    speed_mode = False
    speed_mode_threshold_ms = 900  # Enable when p95 > this (was 1200)
    # SIMPLIFIED: No stability filter - translate all partials
    partial_min_stable_updates = 1  # Accept immediately (was 2)
    
    # AAA: Dynamic emergency flush interval based on playback rate
    emergency_flush_interval = 5.0  # Base: 5s at 1x speed
    
    # Context window for better translations
    context_buffer: list = []  # Keep recent sentences
    max_context = settings.context_buffer_size
    
    # Timing metrics
    session_start_time = time.perf_counter()
    total_translations = 0
    total_translation_time_ms = 0.0
    stt_to_translate_times: List[float] = []  # AAA: Track STT->translate latency
    
    # AAA: Latency metrics for p50/p95 tracking
    fast_pass_latencies: List[float] = []
    refine_pass_latencies: List[float] = []
    
    # AAA: Detected language from Deepgram (auto-propagate to DeepL)
    detected_language: Optional[str] = None
    
    # AAA: Session-scoped refine cache (only cache final/refine translations)
    session_refine_cache: Dict[str, str] = {}
    
    # AAA: Segment tracking for proper paragraph handling
    current_segment_id = 0  # Increments on each original_complete
    
    # SESSION-LEVEL: Accumulated text (shared between listen_to_deepgram and background_flush)
    accumulated_text = ""
    current_interim_transcript = ""  # AAA: Track interim text for forced flushing
    last_emergency_flush_at = time.perf_counter()
    last_final_at = time.perf_counter()  # AAA: Track last final result time
    
    # x2: Watchdog for stalled Deepgram results
    last_audio_received_at = time.perf_counter()
    last_result_received_at = time.perf_counter()
    last_speech_at = time.perf_counter()  # DEBUG: track last speech event
    heartbeat_task: Optional[asyncio.Task] = None  # DEBUG: 2s heartbeat
    
    # DEBUG: Track first audio chunk for logging
    first_audio_logged = False

    async def safe_send(data: dict) -> bool:
        """Send JSON to client with error handling. Uses lock for serialization."""
        nonlocal ws_closed
        if ws_closed:
            return False
        try:
            async with send_lock:  # Benefit: prevents dropped/out-of-order messages
                await websocket.send_json(data)
            return True
        except Exception as e:
            logger.debug(f"[{session_id}] Send failed: {e}")
            return False

    async def process_and_translate(
        text: str,
        segment_id: int,  # AAA: Bind translation to specific segment
        is_refine_pass: bool = False,
        confidence: float = 1.0,  # AAA: Pass confidence for adaptive syntax fix
        is_boundary: bool = False,  # AAA: True if this is a sentence/segment boundary
    ) -> None:
        """Process text and translate with optional syntax fix.
        
        Args:
            text: Text to translate
            segment_id: The segment ID this translation belongs to
            is_refine_pass: True for speech_final (full sentence), False for is_final (fast)
            confidence: Deepgram confidence score (0-1)
            is_boundary: True for segment boundaries (speech_final, should_flush), False for partials
        """
        nonlocal ws_closed, total_translations, total_translation_time_ms, stt_to_translate_times
        if ws_closed:
            return
        
        translate_start = time.perf_counter()
        processed_text = text
        
        try:
            # AAA: Adaptive Syntax Fix - only if:
            # 1. Refine pass AND
            # 2. Quality mode AND 
            # 3. Syntax fix enabled AND
            # 4. Confidence < adaptive threshold
            adaptive_syntax_threshold = settings.adaptive_syntax_threshold
            should_run_syntax_fix = (
                is_refine_pass
                and quality_mode == "quality"
                and settings.enable_syntax_fix
                and confidence < adaptive_syntax_threshold
            )
            
            if should_run_syntax_fix:
                logger.debug(f"[{session_id}] Running syntax fix (confidence={confidence:.2f} < {adaptive_syntax_threshold})")
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
            elif is_refine_pass and quality_mode == "quality" and settings.enable_syntax_fix:
                logger.debug(f"[{session_id}] Skipping syntax fix (confidence={confidence:.2f} >= {adaptive_syntax_threshold})")
            
            # Check for cancellation before translation
            if asyncio.current_task().cancelled():
                return
            
            # AAA: Session-scoped refine cache - only for refine pass
            cache_key = processed_text.lower().strip()
            if is_refine_pass and cache_key in session_refine_cache:
                translation = session_refine_cache[cache_key]
                logger.debug(f"[{session_id}] Session cache hit for refine")
            else:
                # AAA: Use detected language from Deepgram if available (faster DeepL)
                effective_source = source_lang
                if (effective_source == "auto" or not effective_source) and detected_language:
                    effective_source = detected_language
                    logger.debug(f"[{session_id}] Using Deepgram detected language: {detected_language}")
                
                # AAA: Different timeouts for fast vs refine
                timeout = settings.translation_timeout_fast_ms if not is_refine_pass else settings.translation_timeout_ms
                
                # Translate with source language hint
                translation = await translation_service.translate_text(
                    text=processed_text,
                    target_lang=target_lang,
                    source_lang=effective_source if effective_source != "auto" else None,
                    provider=translation_provider,
                    timeout_ms=timeout,
                    is_refine=is_refine_pass,  # AAA: Pass refine flag for quality tuning
                )
                
                # AAA: Cache only refine pass results (session-scoped)
                if is_refine_pass and translation:
                    session_refine_cache[cache_key] = translation
                    # Keep cache bounded
                    if len(session_refine_cache) > 100:
                        # Remove oldest entries
                        oldest_keys = list(session_refine_cache.keys())[:20]
                        for k in oldest_keys:
                            session_refine_cache.pop(k, None)
            
            # Check for cancellation after translation
            if asyncio.current_task().cancelled():
                return
            
            translate_duration_ms = (time.perf_counter() - translate_start) * 1000
            total_translations += 1
            total_translation_time_ms += translate_duration_ms
            
            # AAA: Log translation timing (redacted - no user text for privacy)
            pass_type = "refine" if is_refine_pass else "fast"
            logger.info(f"[{session_id}] Translation {pass_type}: {translate_duration_ms:.0f}ms, chars={len(processed_text)}")
            
            # AAA: Track latency by pass type for p50/p95 metrics
            if is_refine_pass:
                refine_pass_latencies.append(translate_duration_ms)
            else:
                fast_pass_latencies.append(translate_duration_ms)
            
            # Send translation with metadata
            # AAA: final=True at boundaries (speech_final, should_flush, refine) OR Speed Mode boundary
            is_final = is_refine_pass or is_boundary  # Benefit: segments always close properly at 1.5x-2x
            
            # Change 4: Skip empty translations - Benefit: draft never disappears unexpectedly
            if not translation or not translation.strip():
                logger.debug(f"[{session_id}] Skipping empty translation for seg={segment_id}")
                return
            
            # Change 10: Log boundary translation as final=true - Benefit: easy QA verification
            if is_boundary:
                logger.info(f"[{session_id}] BOUNDARY_TX: seg={segment_id}, final=true, chars={len(translation)}")
            
            # DEBUG: Log translation send
            if DEBUG_STALLS:
                logger.info(f"[{session_id}] DEBUG_TX: {{\"seg\":{segment_id},\"pass\":\"{pass_type}\",\"final\":{str(is_final).lower()},\"lat_ms\":{translate_duration_ms:.0f},\"len\":{len(translation)}}}")
            
            await safe_send({
                "type": "translation",
                "original": processed_text,
                "translation": translation,
                "is_refine": is_refine_pass,  # Optional field for HUD
                "final": is_final,  # HUD should commit to history when True
                "segment_id": segment_id,  # For paragraph tracking
                "latency_ms": round(translate_duration_ms, 1),  # Optional timing
                "detected_lang": detected_language,  # AAA: Send detected language to UI
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

    async def _schedule_now(text: str, segment_id: int, is_refine: bool, confidence: float = 1.0, is_boundary: bool = False) -> None:
        """Execute translation immediately with latest-wins cancellation.
        
        Also handles backpressure and Speed Mode for x2 playback.
        """
        nonlocal current_fast_task, current_refine_task, last_translate_scheduled_at
        nonlocal backpressure_engaged, effective_min_partial_chars, speed_mode
        
        last_translate_scheduled_at = time.perf_counter()
        
        # AAA: Speed Mode + Backpressure check
        if len(fast_pass_latencies) >= 5:
            sorted_lat = sorted(fast_pass_latencies[-20:])  # Last 20 calls
            p95_idx = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)
            current_p95 = sorted_lat[p95_idx]
            
            # Speed Mode: Enable when p95 is very high (x2 playback scenario)
            if current_p95 > speed_mode_threshold_ms:
                if not speed_mode:
                    speed_mode = True
                    logger.warning(f"[{session_id}] Speed Mode ON! p95={current_p95:.1f}ms > {speed_mode_threshold_ms}ms")
                    # DEBUG: Log speed mode toggle
                    if DEBUG_STALLS:
                        logger.info(f"[{session_id}] DEBUG_SM: {{\"enabled\":true,\"p95\":{current_p95:.0f}}}")
                    await safe_send({"type": "speed_mode", "enabled": True})
            elif speed_mode and current_p95 < speed_mode_threshold_ms * 0.6:
                speed_mode = False
                logger.info(f"[{session_id}] Speed Mode OFF, p95={current_p95:.1f}ms")
                # DEBUG: Log speed mode toggle
                if DEBUG_STALLS:
                    logger.info(f"[{session_id}] DEBUG_SM: {{\"enabled\":false,\"p95\":{current_p95:.0f}}}")
                await safe_send({"type": "speed_mode", "enabled": False})
            
            # Backpressure: Reduce partial translation frequency
            if current_p95 > settings.backpressure_p95_threshold_ms:
                if not backpressure_engaged:
                    backpressure_engaged = True
                    new_min = int(settings.partial_translate_min_chars * settings.backpressure_min_chars_multiplier)
                    logger.warning(
                        f"[{session_id}] Backpressure engaged! p95={current_p95:.1f}ms, "
                        f"min_chars {effective_min_partial_chars} -> {new_min}"
                    )
                    effective_min_partial_chars = new_min
            elif backpressure_engaged and current_p95 < settings.backpressure_p95_threshold_ms * 0.7:
                backpressure_engaged = False
                effective_min_partial_chars = settings.partial_translate_min_chars
                logger.info(f"[{session_id}] Backpressure released, p95={current_p95:.1f}ms")
        
        # AAA: In Speed Mode, skip refine ONLY for non-boundary (partials already have fast preview)
        # Benefit: boundary translations always get final=true even in Speed Mode
        if speed_mode and is_refine and not is_boundary:
            logger.debug(f"[{session_id}] Skipping non-boundary refine (Speed Mode active)")
            return
        
        async with translate_task_lock:
            if is_refine:
                # REFINE: Run in separate task, NEVER cancel existing refine
                # Just fire and forget - let it complete in background
                current_refine_task = asyncio.create_task(
                    process_and_translate(text, segment_id, is_refine_pass=True, confidence=confidence, is_boundary=is_boundary)
                )
                logger.debug(f"[{session_id}] Refine task started (will complete in background)")
            else:
                # FAST: Cancel previous fast task (latest-wins)
                if current_fast_task and not current_fast_task.done():
                    current_fast_task.cancel()
                    try:
                        await current_fast_task
                    except asyncio.CancelledError:
                        pass
                
                # Create new fast task
                current_fast_task = asyncio.create_task(
                    process_and_translate(text, segment_id, is_refine_pass=False, confidence=confidence, is_boundary=is_boundary)
                )

    async def schedule_translation(text: str, segment_id: int, is_refine: bool, confidence: float = 1.0, is_boundary: bool = False) -> None:
        """Schedule translation with debounce for fast pass, immediate for refine.
        
        Also filters out partial translations if too short.
        
        Args:
            segment_id: The segment ID this translation belongs to
            is_boundary: True for segment boundaries (speech_final, should_flush), marks final in Speed Mode
        """
        nonlocal pending_fast_task, context_buffer
        
        # DEBUG: Log every schedule attempt
        if DEBUG_STALLS:
            logger.info(f"[{session_id}] DEBUG_SCHED: {{\"refine\":{str(is_refine).lower()},\"boundary\":{str(is_boundary).lower()},\"len\":{len(text)},\"txt\":\"{text[:30]}...\"}}")
        
        # Filter: Skip partials that are too short - use unified threshold
        # Benefit: predictable partial flow, fewer stalls
        if not is_refine and len(text.strip()) < effective_min_partial_chars:
            logger.debug(f"[{session_id}] DROPPED short partial ({len(text)} chars < {effective_min_partial_chars})")
            return
        
        # Add to context buffer if refine (final sentence)
        if is_refine:
            context_buffer.append(text)
            if len(context_buffer) > max_context:
                context_buffer.pop(0)
        
        min_interval = settings.translation_rate_limit_ms / 1000.0
        
        # Debounce FAST pass only (refine must always run immediately)
        # Note: Boundary fast passes (should_flush before speech_final) are NOT debounced
        if not is_refine and not is_boundary and min_interval > 0:
            # Cancel any pending fast task
            if pending_fast_task and not pending_fast_task.done():
                pending_fast_task.cancel()
                try:
                    await pending_fast_task
                except asyncio.CancelledError:
                    pass
            
            async def delayed() -> None:
                delay = max(0.0, min_interval - (time.perf_counter() - last_translate_scheduled_at))
                if delay > 0:
                    await asyncio.sleep(delay)
                await _schedule_now(text, segment_id, is_refine=False, confidence=confidence, is_boundary=False)
            
            pending_fast_task = asyncio.create_task(delayed())
            return
        
        # If refine arrives, cancel any pending fast preview (refine is better)
        # AAA: Reviewer fix - Don't cancel refine when fast arrives, but DO cancel pending fast if refine arrives
        if is_refine and pending_fast_task and not pending_fast_task.done():
            pending_fast_task.cancel()
        
        await _schedule_now(text, segment_id, is_refine=is_refine, confidence=confidence, is_boundary=is_boundary)

    async def listen_to_deepgram() -> None:
        """Listen for Deepgram responses with two-pass translation, VAD events, and detected language."""
        nonlocal current_sentence, ws_closed, detected_language, last_partial_translate_time
        nonlocal last_audio_received_at, last_result_received_at, current_segment_id
        nonlocal accumulated_text, current_interim_transcript, last_emergency_flush_at, last_final_at  # Session-level shared variables
        
        last_translated_text = ""  # Avoid re-translating same text
        first_result_logged = False  # AAA: Debug - log first result
        
        # TIME-BASED FLUSH: If partial grows but no final for >3s, force translate
        # last_final_at is now session-scoped
        time_flush_interval = 3.0  # Force translation every 3s if no final
        
        try:
            async for message in deepgram_ws:
                if ws_closed:
                    break
                    
                # x2: Watchdog - check for stalled Deepgram results (soft-warn only)
                now = time.perf_counter()
                
                # NOTE: Emergency flush is now handled by background_flush task (runs every 1s independently)
                
                if now - last_audio_received_at < 5.0:  # Audio still coming
                    if now - last_result_received_at > 8.0:  # But no results >8s
                        logger.warning(f"[{session_id}] Deepgram may be stalled (audio active, no results for {now - last_result_received_at:.1f}s)")
                        # Soft-warn only: send warning to client but don't disconnect
                        # Traffic may resume; keep connection alive
                        await safe_send({"type": "warning", "message": "STT response delayed"})
                        # Reset timestamp to avoid spam
                        last_result_received_at = now
                
                try:
                    data = json.loads(message)
                    
                    # Update result timestamp on any valid message
                    last_result_received_at = now
                    
                    # AAA: Handle VAD events (speech start/end)
                    if data.get("type") == "SpeechStarted":
                        logger.debug(f"[{session_id}] Speech started")
                        await safe_send({"type": "speech_started"})
                        continue
                    
                    # AAA: Handle UtteranceEnd event (more reliable than speech_final in some cases)
                    if data.get("type") == "UtteranceEnd":
                        logger.debug(f"[{session_id}] Utterance end detected")
                        # If we have accumulated text, treat as sentence end
                        if accumulated_text.strip():
                            async with segment_lock:  # Change 2: Benefit: stable segment order
                                final_text = accumulated_text.strip()
                                accumulated_text = ""
                                completed_segment_id = current_segment_id
                                current_segment_id += 1
                            
                            # DEBUG: Log original_complete
                            if DEBUG_STALLS:
                                logger.info(f"[{session_id}] DEBUG_OC: {{\"seg\":{completed_segment_id},\"len\":{len(final_text)},\"words\":{len(final_text.split())},\"reason\":\"utterance_end\"}}")
                            
                            await safe_send({
                                "type": "original_complete",
                                "text": final_text,
                                "segment_id": completed_segment_id,
                            })
                            # FAST PASS at boundary (backup if refine fails) - Benefit: always final commit
                            await schedule_translation(final_text, completed_segment_id, is_refine=False, confidence=0.9, is_boundary=True)
                            # is_boundary=True: SpeechEnded is a segment boundary
                            await schedule_translation(final_text, completed_segment_id, is_refine=True, confidence=0.9, is_boundary=True)
                        continue
                    
                    if data.get("type") == "Results":
                        channel = data.get("channel", {})
                        alternatives = channel.get("alternatives", [])
                        
                        # AAA: Log first Results for debugging
                        if not first_result_logged:
                            first_result_logged = True
                            logger.info(f"[{session_id}] First Deepgram Results received")
                        
                        # AAA: Extract detected language from Deepgram response
                        metadata = data.get("metadata", {})
                        if metadata.get("detected_language") and not detected_language:
                            detected_language = metadata["detected_language"]
                            logger.info(f"[{session_id}] Deepgram detected language: {detected_language}")
                            await safe_send({
                                "type": "detected_language",
                                "language": detected_language,
                            })
                        
                        # Also check model_info for language
                        if not detected_language:
                            model_info = data.get("model_info", {})
                            if model_info.get("language"):
                                detected_language = model_info["language"]
                        
                        if alternatives:
                            alt = alternatives[0]
                            transcript = alt.get("transcript", "")
                            confidence = alt.get("confidence", 1.0)
                            is_final = data.get("is_final", False)
                            speech_final = data.get("speech_final", False)
                            
                            # DEBUG: Log every result from Deepgram
                            if DEBUG_STALLS:
                                logger.info(f"[{session_id}] DEBUG_DG: {{\"final\":{str(is_final).lower()},\"speech\":{str(speech_final).lower()},\"conf\":{confidence:.2f},\"len\":{len(transcript)},\"txt\":\"{transcript[:30]}...\"}}")
                            
                            # === CONFIDENCE FILTERING ===
                            # DISABLED: Deepgram sends confidence=0 for interim results, which breaks filtering
                            # Only filter FINAL results with very low confidence
                            if settings.enable_confidence_filter and is_final and confidence > 0:
                                threshold = settings.min_confidence_threshold_speed if speed_mode else settings.min_confidence_threshold
                                if confidence < threshold:
                                    logger.warning(f"[{session_id}] DROPPED low confidence final ({confidence:.2f} < {threshold:.2f}): {transcript[:50]}")
                                    continue
                            
                            if transcript:
                                if is_final:
                                    # Accumulate finalized text
                                    async with segment_lock:  # Benefit: stable flow without lost segments
                                        accumulated_text += transcript + " "
                                        current_sentence = accumulated_text.strip()
                                        current_interim_transcript = ""  # Clear interim on final
                                    
                                    # Send partial update
                                    await safe_send({
                                        "type": "partial",
                                        "text": current_sentence,
                                    })
                                    
                                    # AAA: Check for sentence-ending punctuation or max words
                                    # This forces earlier translation for long sentences
                                    word_count = len(current_sentence.split())
                                    has_sentence_end = any(current_sentence.rstrip().endswith(p) for p in ['.', '!', '?', '。', '！', '？'])
                                    max_words = settings.max_words_before_flush
                                    should_flush = has_sentence_end or (word_count >= max_words)
                                    
                                    # === TWO-PASS TRANSLATION ===
                                    if settings.enable_two_pass_translation:
                                        # FAST PASS: Translate immediately on is_final
                                        # Only if text changed significantly AND has enough words
                                        # is_boundary=False: mid-sentence fast preview
                                        if (
                                            not speech_final 
                                            and not should_flush
                                            and word_count >= settings.min_words_for_translation
                                            and current_sentence != last_translated_text
                                        ):
                                            last_translated_text = current_sentence
                                            await schedule_translation(
                                                current_sentence, 
                                                current_segment_id,
                                                is_refine=False,
                                                confidence=confidence,
                                                is_boundary=False  # Fast pass mid-sentence = NOT final
                                            )
                                    
                                    # AAA: Force flush on punctuation or max words (treat as speech_final)
                                    if should_flush and not speech_final:
                                        logger.debug(f"[{session_id}] Auto-flush: {word_count} words, punct={has_sentence_end}")
                                        async with segment_lock:  # Change 2: Benefit: stable segment order
                                            final_text = accumulated_text.strip()
                                            accumulated_text = ""
                                            completed_segment_id = current_segment_id
                                            current_segment_id += 1
                                        last_final_at = time.perf_counter()  # Reset time-based flush timer
                                        last_emergency_flush_at = time.perf_counter()  # Reset emergency flush timer
                                        
                                        # DEBUG: Log original_complete
                                        if DEBUG_STALLS:
                                            logger.info(f"[{session_id}] DEBUG_OC: {{\"seg\":{completed_segment_id},\"len\":{len(final_text)},\"words\":{word_count},\"reason\":\"should_flush\"}}")
                                        
                                        await safe_send({
                                            "type": "original_complete",
                                            "text": final_text,
                                            "segment_id": completed_segment_id,
                                        })
                                        
                                        # FAST PASS at boundary (backup if refine fails) - Benefit: always final commit
                                        await schedule_translation(
                                            final_text, 
                                            completed_segment_id,
                                            is_refine=False,
                                            confidence=confidence,
                                            is_boundary=True
                                        )
                                        # REFINE PASS for auto-flushed sentence
                                        # is_boundary=True: should_flush is a segment boundary
                                        await schedule_translation(
                                            final_text, 
                                            completed_segment_id,
                                            is_refine=True,
                                            confidence=confidence,
                                            is_boundary=True
                                        )
                                    elif speech_final:
                                        # End of utterance - send complete original
                                        async with segment_lock:  # Change 2: Benefit: stable segment order
                                            final_text = accumulated_text.strip()
                                            accumulated_text = ""
                                            completed_segment_id = current_segment_id
                                            current_segment_id += 1
                                        last_final_at = time.perf_counter()  # Reset time-based flush timer
                                        last_emergency_flush_at = time.perf_counter()  # Reset emergency flush timer
                                        
                                        # DEBUG: Log original_complete
                                        if DEBUG_STALLS:
                                            logger.info(f"[{session_id}] DEBUG_OC: {{\"seg\":{completed_segment_id},\"len\":{len(final_text)},\"words\":{len(final_text.split())},\"reason\":\"speech_final\"}}")
                                        
                                        await safe_send({
                                            "type": "original_complete",
                                            "text": final_text,
                                            "segment_id": completed_segment_id,
                                        })
                                        
                                        # FAST PASS at boundary (backup if refine fails) - Benefit: always final commit
                                        await schedule_translation(
                                            final_text, 
                                            completed_segment_id,
                                            is_refine=False,
                                            confidence=confidence,
                                            is_boundary=True
                                        )
                                        # REFINE PASS: Full sentence translation with confidence
                                        # is_boundary=True: speech_final is a segment boundary
                                        await schedule_translation(
                                            final_text, 
                                            completed_segment_id,
                                            is_refine=True,
                                            confidence=confidence,
                                            is_boundary=True
                                        )
                                else:
                                    # Interim result - show preview
                                    preview = accumulated_text + transcript
                                    async with segment_lock:  # Benefit: stable flow without lost segments
                                        current_interim_transcript = preview.strip()  # Track interim for forced flush
                                    await safe_send({
                                        "type": "partial",
                                        "text": preview.strip(),
                                    })
                                    
                                    # SIMPLIFIED: Translate partials more aggressively
                                    preview_text = preview.strip()
                                    
                                    # Fast translation from partials (throttled, no stability filter)
                                    now = time.perf_counter()
                                    partial_interval = settings.partial_translate_interval_ms / 1000.0
                                    time_since_last = now - last_partial_translate_time
                                    time_since_final = now - last_final_at
                                    
                                    # TIME-BASED FLUSH: Force translate if no final for >3s
                                    force_translate = time_since_final > time_flush_interval
                                    
                                    # DEBUG: Log interim result processing
                                    if DEBUG_STALLS:
                                        should_translate = (
                                            len(preview_text) >= effective_min_partial_chars
                                            and time_since_last >= partial_interval
                                            and preview_text != last_translated_text
                                        ) or (force_translate and len(preview_text) >= 5)
                                        logger.info(f"[{session_id}] DEBUG_INTERIM: {{\"len\":{len(preview_text)},\"minChars\":{effective_min_partial_chars},\"timeSinceLast\":{time_since_last*1000:.0f}ms,\"timeSinceFinal\":{time_since_final:.1f}s,\"force\":{str(force_translate).lower()},\"will_translate\":{str(should_translate).lower()}}}")
                                    
                                    if (
                                        len(preview_text) >= effective_min_partial_chars
                                        and time_since_last >= partial_interval
                                        and preview_text != last_translated_text
                                    ) or (force_translate and len(preview_text) >= 5):
                                        last_partial_translate_time = now
                                        last_translated_text = preview_text
                                        if force_translate:
                                            logger.info(f"[{session_id}] Time-forced partial: {len(preview_text)} chars ({time_since_final:.1f}s since final)")
                                        else:
                                            logger.info(f"[{session_id}] Partial accepted: {len(preview_text)} chars")
                                        # is_boundary=False: partial is NOT a segment boundary
                                        await schedule_translation(
                                            preview_text,
                                            current_segment_id,
                                            is_refine=False,
                                            confidence=confidence,
                                            is_boundary=False
                                        )
                                        
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            if not ws_closed:
                logger.error(f"[{session_id}] Deepgram listener error: {e}")

    # === AAA: DEEPGRAM LANGUAGE HINT ===
    def get_deepgram_language_code(lang: str) -> Optional[str]:
        """Map source language to Deepgram language parameter."""
        if not lang or lang == "auto":
            return None
        
        # Deepgram language codes (Nova-2/Nova-3 supported)
        lang_map = {
            "en": "en", "en-US": "en-US", "en-GB": "en-GB",
            "es": "es", "fr": "fr", "de": "de", "it": "it",
            "pt": "pt", "pt-BR": "pt-BR", "nl": "nl",
            "ja": "ja", "zh": "zh", "ko": "ko", "ru": "ru",
            "pl": "pl", "tr": "tr", "uk": "uk", "el": "el",
            "cs": "cs", "da": "da", "fi": "fi", "hi": "hi",
            "id": "id", "no": "no", "sv": "sv", "th": "th", "vi": "vi",
        }
        return lang_map.get(lang.lower(), lang.lower())

    async def connect_to_deepgram(sample_rate: int = 16000) -> None:
        """Connect to Deepgram with language hint, VAD events, and proper endpointing."""
        nonlocal deepgram_ws, listen_task
        
        # Use configurable model (nova-2 or nova-3)
        # For better music/singing recognition, we use specific settings:
        # - No diarization (single speaker focus)
        # - Keywords boost (optional)
        # - Filler words enabled
        params = [
            f"model={settings.deepgram_model}",
            "encoding=linear16",
            f"sample_rate={sample_rate}",
            "channels=1",
            "punctuate=true",
            "interim_results=true",
            # AAA: Enable smart formatting for cleaner output
            "smart_format=true",
            # AAA: Filler words help with singing/humming detection
            "filler_words=true",
            # AAA: Utterances mode for better sentence boundary detection
            "utterances=true",
            # AAA: Disable diarization for faster processing and better music handling
            "diarize=false",
        ]
        
        # AAA: Enable VAD events for speech start/end detection (if configured)
        if settings.deepgram_vad_events:
            params.append("vad_events=true")
        
        # AAA: Adaptive endpointing per mode - use client value if sent, else fall back to config
        # client_utterance_end_ms is set from config message (fast=280ms, quality=600ms)
        endpointing_ms = client_utterance_end_ms if client_utterance_end_ms > 0 else settings.deepgram_utterance_end_ms
        if endpointing_ms > 0:
            params.append(f"endpointing={endpointing_ms}")
            logger.info(f"[{session_id}] Using endpointing={endpointing_ms}ms (mode={quality_mode})")
        
        # AAA: Add language hint if source language is known (major speed win!)
        dg_lang = get_deepgram_language_code(source_lang)
        if dg_lang:
            params.append(f"language={dg_lang}")
            logger.info(f"[{session_id}] Using Deepgram language hint: {dg_lang}")
        else:
            logger.info(f"[{session_id}] No language hint - Deepgram will auto-detect")
        
        url = f"{DEEPGRAM_URL}?{'&'.join(params)}"
        logger.info(f"[{session_id}] Deepgram URL: {url}")
        logger.info(f"[{session_id}] API key present: {bool(deepgram_api_key)}")
        
        deepgram_ws = await websockets.connect(
            url,
            additional_headers={"Authorization": f"Token {deepgram_api_key}"},
            ping_interval=20,
            ping_timeout=10,
        )
        
        logger.info(f"[{session_id}] Connected to Deepgram!")
        listen_task = asyncio.create_task(listen_to_deepgram())
        
        # AAA: Keepalive task - send silent audio to Deepgram every 5s if no audio received
        # This prevents Deepgram from timing out during pauses/silence
        async def deepgram_keepalive():
            nonlocal last_audio_received_at, keepalive_task
            # 100ms of silence at 16kHz = 1600 samples * 2 bytes = 3200 bytes
            silent_audio = bytes(3200)
            keepalive_count = 0
            while not ws_closed and deepgram_ws:
                await asyncio.sleep(3.0)  # Check every 3 seconds
                if ws_closed or not deepgram_ws:
                    break
                # If no audio for >2 seconds, send keepalive silence
                time_since_audio = time.perf_counter() - last_audio_received_at
                if time_since_audio > 2.0:
                    keepalive_count += 1
                    try:
                        await deepgram_ws.send(silent_audio)
                        logger.info(f"[{session_id}] Keepalive #{keepalive_count}: sent silent audio (no audio for {time_since_audio:.1f}s)")
                    except Exception as e:
                        logger.warning(f"[{session_id}] Keepalive failed: {e}")
                        break
        
        keepalive_task = asyncio.create_task(deepgram_keepalive())
        
        # BACKGROUND FLUSH TASK: Force flush accumulated text if no speech_final
        # This runs independently from the Deepgram message loop!
        # Uses emergency_flush_interval which adapts to playback rate
        async def background_flush():
            nonlocal accumulated_text, current_segment_id, last_emergency_flush_at, last_speech_at
            nonlocal current_interim_transcript, last_final_at
            tick_count = 0
            last_activity_warning_at = 0.0
            while not ws_closed:
                await asyncio.sleep(1.0)  # Check every 1 second
                if ws_closed:
                    break
                
                tick_count += 1
                now = time.perf_counter()
                time_since_flush = now - last_emergency_flush_at
                time_since_speech = now - last_speech_at
                time_since_audio = now - last_audio_received_at
                time_since_result = now - last_result_received_at
                acc_len = len(accumulated_text.strip())
                
                # DEBUG: Log background flush status every 5 ticks
                if DEBUG_STALLS and tick_count % 5 == 0:
                    logger.info(f"[{session_id}] DEBUG_BG: {{\"tick\":{tick_count},\"acc_len\":{acc_len},\"flush_int\":{emergency_flush_interval:.0f}s,\"since_flush\":{time_since_flush:.1f}s,\"since_result\":{time_since_result:.1f}s}}")
                
                # MUSIC DETECTION: If we're receiving audio but no results for >10s, warn user
                if time_since_audio < 2.0 and time_since_result > 10.0 and (now - last_activity_warning_at) > 30.0:
                    last_activity_warning_at = now
                    logger.warning(f"[{session_id}] MUSIC DETECTED: Audio flowing but no transcripts for {time_since_result:.1f}s")
                    await safe_send({
                        "type": "info",
                        "message": "Music/singing detected - waiting for speech...",
                    })
                
                # If we have accumulated text and no flush for >emergency_flush_interval, force it
                time_since_final = now - last_final_at
                interim_len = len(current_interim_transcript)
                
                # Condition 1: Accumulated text needs flushing (standard case)
                should_flush_accumulated = acc_len > 0 and time_since_flush > emergency_flush_interval
                
                # Condition 2: Stalled interim result (singing/music case)
                # If we have interim text but no final for >5s, force finalize it
                should_flush_interim = acc_len == 0 and interim_len > 0 and time_since_final > 5.0
                
                if should_flush_accumulated or should_flush_interim:
                    last_emergency_flush_at = now
                    last_final_at = now  # Reset final timer too
                    
                    async with segment_lock:  # Change 2: Benefit: stable segment order
                        completed_segment_id = current_segment_id
                        
                        if should_flush_interim:
                            final_text = current_interim_transcript.strip()
                            current_interim_transcript = ""
                        else:
                            final_text = accumulated_text.strip()
                            accumulated_text = ""
                        
                        current_segment_id += 1
                    
                    if should_flush_interim:
                        logger.warning(f"[{session_id}] BACKGROUND FLUSH: Forcing STALLED INTERIM segment {completed_segment_id} ({len(final_text)} chars)")
                    else:
                        logger.warning(f"[{session_id}] BACKGROUND FLUSH: Forcing segment {completed_segment_id} ({len(final_text)} chars)")
                    
                    if DEBUG_STALLS:
                        logger.info(f"[{session_id}] DEBUG_OC: {{\"seg\":{completed_segment_id},\"len\":{len(final_text)},\"reason\":\"background_flush\"}}")
                    
                    await safe_send({
                        "type": "original_complete",
                        "text": final_text,
                        "segment_id": completed_segment_id,
                    })
                    # FAST PASS at boundary (backup if refine fails) - Benefit: always final commit
                    await schedule_translation(final_text, completed_segment_id, is_refine=False, confidence=0.8, is_boundary=True)
                    await schedule_translation(final_text, completed_segment_id, is_refine=True, confidence=0.8, is_boundary=True)
        
        flush_task = asyncio.create_task(background_flush())

    try:
        # AAA: Wait for config before connecting to Deepgram (for language hint)
        audio_buffer: list = []
        
        await safe_send({
            "type": "waiting_config",
            "message": "Waiting for language configuration...",
            "session_id": session_id,
        })

        # Handle client messages
        first_audio_chunk_logged = False  # Local flag for logging
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
                            source_lang = data.get("sourceLang", None)
                            translation_provider = data.get("translationProvider", "deepl")
                            quality_mode = data.get("qualityMode", "fast")
                            # Get sample rate from client (HUD sends actual audioContext.sampleRate)
                            client_sample_rate = data.get("sampleRate", 16000)
                            # Get playback rate from client (video playback speed)
                            playback_rate = data.get("playbackRate", 1.0)
                            
                            # AAA: Adjust thresholds based on playback rate
                            # Speed Mode: >= 1.5x (fast videos need aggressive flushing)
                            # Quality Mode: <= 0.75x (slow videos can afford longer processing)
                            if playback_rate >= 1.5:
                                emergency_flush_interval = 3.0  # Faster flush at 1.5x-2x
                                speed_mode = True  # Force speed mode
                                logger.info(f"[{session_id}] Speed Mode enabled (playback={playback_rate}x): flush=3s")
                            elif playback_rate <= 0.75:
                                emergency_flush_interval = 7.0  # Longer flush at 0.5x-0.75x
                                speed_mode = False
                                logger.info(f"[{session_id}] Quality Mode (playback={playback_rate}x): flush=7s")
                            else:
                                emergency_flush_interval = 5.0  # Default at 1x
                            
                            # AAA: Determine endpointing based on qualityMode (backend decides, not client)
                            if quality_mode == "quality":
                                client_utterance_end_ms = settings.deepgram_utterance_end_quality_ms
                            else:
                                client_utterance_end_ms = settings.deepgram_utterance_end_fast_ms
                            
                            logger.info(
                                f"[{session_id}] Config: {source_lang} -> {target_lang}, "
                                f"provider={translation_provider}, quality={quality_mode}, "
                                f"sampleRate={client_sample_rate}, playbackRate={playback_rate}x, "
                                f"endpointing={client_utterance_end_ms}ms, flushInterval={emergency_flush_interval}s"
                            )
                            
                            # AAA: Connect to Deepgram with language hint and sample rate
                            if not config_received:
                                config_received = True
                                await connect_to_deepgram(sample_rate=client_sample_rate)
                                
                                # DEBUG: Log session ready with full config
                                if DEBUG_STALLS:
                                    logger.info(f"[{session_id}] DEBUG_READY: {{\"source\":\"{source_lang}\",\"target\":\"{target_lang}\",\"sampleRate\":{client_sample_rate},\"quality\":\"{quality_mode}\"}}")
                                
                                await safe_send({
                                    "type": "ready",
                                    "message": "Deepgram transcription ready",
                                    "session_id": session_id,
                                    "language_hint": get_deepgram_language_code(source_lang),
                                })
                                
                                # DEBUG: Start 2s heartbeat for diagnostics
                                async def debug_heartbeat():
                                    nonlocal speed_mode, backpressure_engaged, pending_fast_task, total_translations
                                    while not ws_closed:
                                        await asyncio.sleep(2.0)
                                        if DEBUG_STALLS and not ws_closed:
                                            now = time.perf_counter()
                                            fast_p95 = 0.0
                                            if len(fast_pass_latencies) >= 5:
                                                sorted_lat = sorted(fast_pass_latencies[-20:])
                                                p95_idx = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)
                                                fast_p95 = sorted_lat[p95_idx]
                                            pending = pending_fast_task is not None and not pending_fast_task.done()
                                            logger.info(f"[{session_id}] DEBUG_HB: {{\"audio_ms\":{int((now-last_audio_received_at)*1000)},\"result_ms\":{int((now-last_result_received_at)*1000)},\"speech_ms\":{int((now-last_speech_at)*1000)},\"speed\":{speed_mode},\"bp\":{backpressure_engaged},\"pending_fast\":{pending},\"fast_p95\":{fast_p95:.0f},\"translations\":{total_translations}}}")
                                
                                heartbeat_task = asyncio.create_task(debug_heartbeat())
                                
                                # Send buffered audio
                                for buffered_audio in audio_buffer:
                                    if deepgram_ws:
                                        await deepgram_ws.send(buffered_audio)
                                audio_buffer.clear()
                        
                        elif msg_type == "playback_rate_update":
                            # Handle dynamic playback rate changes
                            new_rate = data.get("playbackRate", 1.0)
                            if new_rate != playback_rate:
                                playback_rate = new_rate
                                
                                # Adjust thresholds dynamically
                                if playback_rate >= 1.5:
                                    emergency_flush_interval = 3.0
                                    speed_mode = True
                                    logger.info(f"[{session_id}] Speed Mode activated (playback={playback_rate}x): flush=3s")
                                elif playback_rate <= 0.75:
                                    emergency_flush_interval = 7.0
                                    speed_mode = False
                                    logger.info(f"[{session_id}] Quality Mode activated (playback={playback_rate}x): flush=7s")
                                else:
                                    emergency_flush_interval = 5.0
                                    speed_mode = False
                                    logger.info(f"[{session_id}] Normal Mode (playback={playback_rate}x): flush=5s")
                        
                        elif msg_type == "audio":
                            # x2: Update watchdog timestamp for received audio
                            last_audio_received_at = time.perf_counter()
                            
                            audio_b64 = data.get("data", "")
                            if audio_b64:
                                # AAA: Safe base64 decoding - don't crash on bad data
                                try:
                                    audio_bytes = base64.b64decode(audio_b64)
                                    # Log first audio chunk
                                    if not first_audio_chunk_logged:
                                        first_audio_chunk_logged = True
                                        logger.info(f"[{session_id}] First audio chunk (base64): {len(audio_bytes)} bytes")
                                    if deepgram_ws:
                                        await deepgram_ws.send(audio_bytes)
                                    else:
                                        # Buffer audio until Deepgram connects
                                        audio_buffer.append(audio_bytes)
                                        if len(audio_buffer) > 100:
                                            audio_buffer.pop(0)
                                except Exception as decode_error:
                                    logger.warning(f"[{session_id}] Base64 decode error (skipping chunk): {decode_error}")
                                    # Continue processing - don't kill session
                        
                        elif msg_type == "stop":
                            break
                    
                    elif "bytes" in message:
                        # x2: Update watchdog timestamp for binary audio
                        last_audio_received_at = time.perf_counter()
                        
                        # Log first binary audio chunk
                        if not first_audio_chunk_logged:
                            first_audio_chunk_logged = True
                            logger.info(f"[{session_id}] First audio chunk (binary): {len(message['bytes'])} bytes")
                        
                        if deepgram_ws:
                            await deepgram_ws.send(message["bytes"])
                        else:
                            audio_buffer.append(message["bytes"])
                            if len(audio_buffer) > 100:
                                audio_buffer.pop(0)

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
        
        # Cancel any pending translation tasks
        if pending_fast_task and not pending_fast_task.done():
            pending_fast_task.cancel()
        if current_fast_task and not current_fast_task.done():
            current_fast_task.cancel()
        if current_refine_task and not current_refine_task.done():
            current_refine_task.cancel()
        
        # Cancel background tasks
        if keepalive_task and not keepalive_task.done():
            keepalive_task.cancel()
        if heartbeat_task and not heartbeat_task.done():
            heartbeat_task.cancel()
        
        # Cancel flush task
        try:
            if 'flush_task' in dir() and flush_task and not flush_task.done():
                flush_task.cancel()
        except NameError:
            pass
        
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
        
        # Log session metrics with p50/p95 latency
        session_duration = time.perf_counter() - session_start_time
        avg_translation = (
            total_translation_time_ms / total_translations 
            if total_translations > 0 else 0
        )
        
        # AAA: Calculate p50/p95 latency metrics
        def calc_percentiles(latencies: List[float]) -> Tuple[float, float]:
            if not latencies:
                return 0.0, 0.0
            sorted_lat = sorted(latencies)
            p50 = sorted_lat[len(sorted_lat) // 2]
            p95_idx = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)
            p95 = sorted_lat[p95_idx]
            return p50, p95
        
        fast_p50, fast_p95 = calc_percentiles(fast_pass_latencies)
        refine_p50, refine_p95 = calc_percentiles(refine_pass_latencies)
        
        logger.info(
            f"[{session_id}] Session ended: {session_duration:.1f}s, "
            f"{total_translations} translations, avg {avg_translation:.1f}ms/translation"
        )
        logger.info(
            f"[{session_id}] Fast pass latency: p50={fast_p50:.1f}ms, p95={fast_p95:.1f}ms ({len(fast_pass_latencies)} calls)"
        )
        logger.info(
            f"[{session_id}] Refine pass latency: p50={refine_p50:.1f}ms, p95={refine_p95:.1f}ms ({len(refine_pass_latencies)} calls)"
        )
        if detected_language:
            logger.info(f"[{session_id}] Detected language: {detected_language}")
