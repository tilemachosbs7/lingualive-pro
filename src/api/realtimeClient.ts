/**
 * =============================================================================
 * REALTIMECLIENT.TS — WebSocket Client for /ws/captions
 * =============================================================================
 *
 * Παρέχει typed WebSocket client για real-time captions και translation.
 *
 * ΧΡΗΣΗ:
 * ------
 *   import { RealtimeClient } from "./api/realtimeClient";
 *   import { loadSettings } from "./storage/settings";
 *
 *   const settings = await loadSettings();
 *   const client = new RealtimeClient({
 *     settings,
 *     qualityMode: "FAST",
 *     onEvent: (event) => {
 *       if (event.type === "final_transcript") {
 *         console.log("Transcript:", event.text);
 *       }
 *     },
 *   });
 *
 *   await client.connect();
 *   await client.startSession({
 *     source_lang: "en",
 *     target_lang: "el",
 *   });
 *
 * PROTOCOL:
 * ---------
 * Ακολουθεί το MessageEnvelope protocol του backend (protocol.py):
 * - session_start → session_started
 * - audio_chunk → transcript (partial/final)
 * - session_stop → session_ended
 *
 * AUTH:
 * -----
 * Χρησιμοποιεί την ίδια λογική με τον backendClient:
 * - Αν υπάρχει devToken → x-dev-token header
 * - Αλλιώς → x-debug-user-id + x-debug-plan headers
 *
 * ΣΗΜΑΝΤΙΚΟ:
 * Browsers δεν επιτρέπουν custom headers σε WebSocket connections.
 * Για dev, χρησιμοποιούμε query params για auth.
 * Σε production, θα χρησιμοποιηθεί background script + webRequest.
 * =============================================================================
 */

import {
  type LinguaLiveSettings,
  type PlanTier,
  type QualityMode,
  getWebSocketUrl,
} from "../storage/settings";

// Re-export types for convenience
export type { PlanTier, QualityMode } from "../storage/settings";

// =============================================================================
// MESSAGE TYPES — Aligned with backend protocol.py
// =============================================================================

/**
 * Message types that can be sent/received via WebSocket.
 */
export type MessageType =
  | "session_start"
  | "session_started"
  | "audio_chunk"
  | "session_stop"
  | "session_ended"
  | "transcript"
  | "error";

/**
 * Error codes from the backend.
 */
export type ErrorCode =
  | "PROVIDER_TIMEOUT"
  | "INVALID_LANGUAGE"
  | "RATE_LIMITED"
  | "INTERNAL_ERROR"
  | "INVALID_AUDIO_FORMAT"
  | "SESSION_TIMEOUT"
  | "AUTH_FAILED"
  | "NO_ACTIVE_SESSION"
  | "QUALITY_NOT_ALLOWED_FOR_PLAN";

/**
 * Generic message envelope matching backend MessageEnvelope.
 */
export interface MessageEnvelope<TPayload = unknown> {
  type: MessageType;
  session_id: string;
  seq?: number | null;
  payload: TPayload;
}

// =============================================================================
// PAYLOAD TYPES — Matching backend Pydantic models
// =============================================================================

/**
 * Payload for session_start message (sent by client).
 */
export interface SessionStartPayload {
  /** User ID (optional, can be inferred from auth) */
  user_id?: string | null;
  /** Plan tier (sent but server uses auth value) */
  plan_tier: PlanTier;
  /** Source language code (e.g. "en", "auto") */
  source_lang: string;
  /** Target language code (e.g. "el") */
  target_lang: string;
  /** Quality mode (FAST or SMART) */
  quality_mode: QualityMode;
  /** Extension version (optional) */
  extension_version?: string | null;
}

/**
 * Payload for session_started message (received from server).
 */
export interface SessionStartedPayload {
  session_id: string;
  accepted_target_lang: string;
  accepted_quality_mode: QualityMode;
  provider_chain: string[];
}

/**
 * Payload for audio_chunk message (sent by client).
 */
export interface AudioChunkPayload {
  /** Base64-encoded audio data */
  chunk_base64: string;
  /** Timestamp in milliseconds from session start */
  timestamp_ms: number;
}

/**
 * Payload for session_stop message (sent by client).
 */
export interface SessionStopPayload {
  /** Reason for stopping (optional) */
  reason?: string | null;
}

/**
 * Payload for session_ended message (received from server).
 */
export interface SessionEndedPayload {
  reason: string;
}

/**
 * Payload for transcript message (received from server).
 */
export interface TranscriptPayload {
  text: string;
  translated_text: string | null;
  is_final: boolean;
  timestamp_ms?: number | null;
}

/**
 * Payload for error message (received from server).
 */
export interface ErrorPayload {
  code: ErrorCode | string;
  message: string;
  retryable: boolean;
}

// =============================================================================
// CLIENT EVENTS — Ergonomic event API for HUD/extension
// =============================================================================

/**
 * Events emitted by RealtimeClient for the HUD/extension to handle.
 */
export type RealtimeClientEvent =
  | { type: "connecting" }
  | { type: "open" }
  | { type: "session_started"; payload: SessionStartedPayload }
  | { type: "partial_transcript"; text: string; translatedText: string | null }
  | {
      type: "final_transcript";
      text: string;
      translatedText: string | null;
      timestampMs: number | null;
    }
  | { type: "session_ended"; reason: string }
  | { type: "quality_not_allowed_for_plan"; qualityMode: QualityMode }
  | { type: "error"; error: Error; errorCode?: ErrorCode | string | null }
  | { type: "closed"; code: number; reason: string };

// =============================================================================
// CLIENT OPTIONS
// =============================================================================

/**
 * Options for creating a RealtimeClient.
 */
export interface RealtimeClientOptions {
  /** Extension settings with backend URL and auth */
  settings: LinguaLiveSettings;
  /** Quality mode for the session */
  qualityMode: QualityMode;
  /** Callback for client events */
  onEvent: (event: RealtimeClientEvent) => void;
  /** Optional callback for raw envelopes (for debugging) */
  onRawEnvelope?: (envelope: MessageEnvelope) => void;
}

// =============================================================================
// REALTIME CLIENT CLASS
// =============================================================================

/**
 * WebSocket client for real-time captions and translation.
 *
 * Lifecycle:
 * 1. Create client with options
 * 2. Call connect() to establish WebSocket connection
 * 3. Call startSession() to begin transcription
 * 4. Call sendAudioChunk() repeatedly with audio data
 * 5. Call stopSession() when done
 * 6. Call close() to disconnect
 */
export class RealtimeClient {
  private readonly options: RealtimeClientOptions;
  private ws: WebSocket | null = null;
  private sessionId: string | null = null;
  private isSessionActive = false;
  private sequenceNumber = 0;

  constructor(options: RealtimeClientOptions) {
    this.options = options;
  }

  // ===========================================================================
  // PUBLIC API
  // ===========================================================================

  /**
   * Connect to the WebSocket server.
   *
   * Note: Browsers don't support custom headers on WebSocket connections.
   * For dev, we pass auth via query params. In production, this would use
   * a background script with webRequest API to inject headers.
   */
  async connect(): Promise<void> {
    if (this.ws) {
      throw new Error("Already connected");
    }

    this.emit({ type: "connecting" });

    const wsUrl = this.buildWebSocketUrl();
    this.ws = new WebSocket(wsUrl);

    return new Promise((resolve, reject) => {
      const ws = this.ws!;

      ws.addEventListener("open", () => {
        this.emit({ type: "open" });
        resolve();
      });

      ws.addEventListener("error", () => {
        this.emit({
          type: "error",
          error: new Error("WebSocket connection error"),
        });
        reject(new Error("WebSocket connection error"));
      });

      ws.addEventListener("close", (event) => {
        this.handleClose(event);
      });

      ws.addEventListener("message", (event) => {
        this.handleMessage(event);
      });
    });
  }

  /**
   * Start a transcription session.
   *
   * @param params - Session parameters (languages)
   */
  async startSession(params: {
    sourceLang: string;
    targetLang: string;
    extensionVersion?: string;
  }): Promise<void> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error("WebSocket not connected");
    }

    if (this.isSessionActive) {
      throw new Error("Session already active");
    }

    // Generate a new session ID
    this.sessionId = crypto.randomUUID();
    this.sequenceNumber = 0;

    const payload: SessionStartPayload = {
      plan_tier: this.options.settings.debugPlan ?? "FREE",
      source_lang: params.sourceLang,
      target_lang: params.targetLang,
      quality_mode: this.options.qualityMode,
      extension_version: params.extensionVersion ?? null,
    };

    this.sendEnvelope({
      type: "session_start",
      session_id: this.sessionId,
      seq: this.sequenceNumber++,
      payload,
    });
  }

  /**
   * Send an audio chunk for transcription.
   *
   * @param data - Audio data (ArrayBuffer or Uint8Array)
   * @param timestampMs - Timestamp in milliseconds from session start
   */
  sendAudioChunk(data: ArrayBuffer | Uint8Array, timestampMs: number): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error("WebSocket not connected");
    }

    if (!this.isSessionActive || !this.sessionId) {
      throw new Error("No active session");
    }

    // Convert to base64
    const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
    const chunkBase64 = this.arrayBufferToBase64(bytes);

    const payload: AudioChunkPayload = {
      chunk_base64: chunkBase64,
      timestamp_ms: timestampMs,
    };

    this.sendEnvelope({
      type: "audio_chunk",
      session_id: this.sessionId,
      seq: this.sequenceNumber++,
      payload,
    });
  }

  /**
   * Stop the current transcription session.
   *
   * @param reason - Optional reason for stopping
   */
  async stopSession(reason?: string): Promise<void> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return; // Already disconnected
    }

    if (!this.isSessionActive || !this.sessionId) {
      return; // No active session
    }

    const payload: SessionStopPayload = {
      reason: reason ?? "user_requested",
    };

    this.sendEnvelope({
      type: "session_stop",
      session_id: this.sessionId,
      seq: this.sequenceNumber++,
      payload,
    });

    this.isSessionActive = false;
  }

  /**
   * Close the WebSocket connection.
   *
   * @param code - Close code (default 1000 = normal)
   * @param reason - Close reason
   */
  close(code: number = 1000, reason: string = "Client closed"): void {
    if (this.ws) {
      this.ws.close(code, reason);
      this.ws = null;
    }
    this.sessionId = null;
    this.isSessionActive = false;
  }

  /**
   * Check if the client is connected.
   */
  get isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Check if a session is currently active.
   */
  get hasActiveSession(): boolean {
    return this.isSessionActive;
  }

  // ===========================================================================
  // PRIVATE METHODS
  // ===========================================================================

  /**
   * Build the WebSocket URL with auth query params.
   *
   * Note: This is a dev-only workaround. Browsers don't allow custom headers
   * on WebSocket connections. In production, we would use a background script
   * with webRequest API to inject headers.
   */
  private buildWebSocketUrl(): string {
    const { settings } = this.options;
    const baseUrl = getWebSocketUrl(settings);
    const url = new URL(baseUrl);

    // Add auth as query params (dev workaround)
    // Priority: dev token > debug headers
    if (settings.devToken) {
      url.searchParams.set("dev_token", settings.devToken);
    } else if (settings.debugUserId) {
      url.searchParams.set("debug_user_id", settings.debugUserId);
      if (settings.debugPlan) {
        url.searchParams.set("debug_plan", settings.debugPlan);
      }
    }

    return url.toString();
  }

  /**
   * Send a message envelope to the server.
   */
  private sendEnvelope(envelope: MessageEnvelope): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error("WebSocket not connected");
    }

    this.ws.send(JSON.stringify(envelope));
  }

  /**
   * Handle incoming WebSocket message.
   */
  private handleMessage(event: MessageEvent): void {
    let envelope: MessageEnvelope;

    try {
      envelope = JSON.parse(event.data as string);
    } catch {
      this.emit({
        type: "error",
        error: new Error("Invalid JSON from server"),
      });
      return;
    }

    // Call raw envelope callback if provided (for debugging)
    this.options.onRawEnvelope?.(envelope);

    // Dispatch by message type
    this.dispatchEnvelope(envelope);
  }

  /**
   * Dispatch envelope to appropriate handler.
   */
  private dispatchEnvelope(envelope: MessageEnvelope): void {
    switch (envelope.type) {
      case "session_started":
        this.handleSessionStarted(envelope.payload as SessionStartedPayload);
        break;

      case "transcript":
        this.handleTranscript(envelope.payload as TranscriptPayload);
        break;

      case "session_ended":
        this.handleSessionEnded(envelope.payload as SessionEndedPayload);
        break;

      case "error":
        this.handleError(envelope.payload as ErrorPayload);
        break;

      default:
        // Ignore unknown message types
        console.warn(
          `[RealtimeClient] Unknown message type: ${envelope.type}`
        );
    }
  }

  /**
   * Handle session_started message.
   */
  private handleSessionStarted(payload: SessionStartedPayload): void {
    this.isSessionActive = true;
    this.emit({ type: "session_started", payload });
  }

  /**
   * Handle transcript message.
   */
  private handleTranscript(payload: TranscriptPayload): void {
    if (payload.is_final) {
      this.emit({
        type: "final_transcript",
        text: payload.text,
        translatedText: payload.translated_text,
        timestampMs: payload.timestamp_ms ?? null,
      });
    } else {
      this.emit({
        type: "partial_transcript",
        text: payload.text,
        translatedText: payload.translated_text,
      });
    }
  }

  /**
   * Handle session_ended message.
   */
  private handleSessionEnded(payload: SessionEndedPayload): void {
    this.isSessionActive = false;
    this.emit({ type: "session_ended", reason: payload.reason });
  }

  /**
   * Handle error message.
   */
  private handleError(payload: ErrorPayload): void {
    // Check for quality_not_allowed_for_plan error (backend sends uppercase)
    if (
      payload.code === "QUALITY_NOT_ALLOWED_FOR_PLAN" ||
      payload.code === "quality_not_allowed_for_plan"
    ) {
      this.emit({
        type: "quality_not_allowed_for_plan",
        qualityMode: this.options.qualityMode,
      });
      return;
    }

    this.emit({
      type: "error",
      error: new Error(payload.message),
      errorCode: payload.code,
    });
  }

  /**
   * Handle WebSocket close event.
   */
  private handleClose(event: CloseEvent): void {
    this.isSessionActive = false;
    this.ws = null;

    this.emit({
      type: "closed",
      code: event.code,
      reason: event.reason || "Connection closed",
    });
  }

  /**
   * Emit an event to the callback.
   */
  private emit(event: RealtimeClientEvent): void {
    try {
      this.options.onEvent(event);
    } catch (err) {
      console.error("[RealtimeClient] Error in event handler:", err);
    }
  }

  /**
   * Convert Uint8Array to base64 string.
   */
  private arrayBufferToBase64(bytes: Uint8Array): string {
    let binary = "";
    for (let i = 0; i < bytes.length; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }
}
