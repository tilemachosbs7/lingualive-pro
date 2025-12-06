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
import { type LinguaLiveSettings, type PlanTier, type QualityMode } from "../storage/settings";
export type { PlanTier, QualityMode } from "../storage/settings";
/**
 * Message types that can be sent/received via WebSocket.
 */
export type MessageType = "session_start" | "session_started" | "audio_chunk" | "session_stop" | "session_ended" | "transcript" | "error";
/**
 * Error codes from the backend.
 */
export type ErrorCode = "PROVIDER_TIMEOUT" | "INVALID_LANGUAGE" | "RATE_LIMITED" | "INTERNAL_ERROR" | "INVALID_AUDIO_FORMAT" | "SESSION_TIMEOUT" | "AUTH_FAILED" | "NO_ACTIVE_SESSION" | "QUALITY_NOT_ALLOWED_FOR_PLAN";
/**
 * Generic message envelope matching backend MessageEnvelope.
 */
export interface MessageEnvelope<TPayload = unknown> {
    type: MessageType;
    session_id: string;
    seq?: number | null;
    payload: TPayload;
}
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
/**
 * Events emitted by RealtimeClient for the HUD/extension to handle.
 */
export type RealtimeClientEvent = {
    type: "connecting";
} | {
    type: "open";
} | {
    type: "session_started";
    payload: SessionStartedPayload;
} | {
    type: "partial_transcript";
    text: string;
    translatedText: string | null;
} | {
    type: "final_transcript";
    text: string;
    translatedText: string | null;
    timestampMs: number | null;
} | {
    type: "session_ended";
    reason: string;
} | {
    type: "quality_not_allowed_for_plan";
    qualityMode: QualityMode;
} | {
    type: "error";
    error: Error;
    errorCode?: ErrorCode | string | null;
} | {
    type: "closed";
    code: number;
    reason: string;
};
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
export declare class RealtimeClient {
    private readonly options;
    private ws;
    private sessionId;
    private isSessionActive;
    private sequenceNumber;
    constructor(options: RealtimeClientOptions);
    /**
     * Connect to the WebSocket server.
     *
     * Note: Browsers don't support custom headers on WebSocket connections.
     * For dev, we pass auth via query params. In production, this would use
     * a background script with webRequest API to inject headers.
     */
    connect(): Promise<void>;
    /**
     * Start a transcription session.
     *
     * @param params - Session parameters (languages)
     */
    startSession(params: {
        sourceLang: string;
        targetLang: string;
        extensionVersion?: string;
    }): Promise<void>;
    /**
     * Send an audio chunk for transcription.
     *
     * @param data - Audio data (ArrayBuffer or Uint8Array)
     * @param timestampMs - Timestamp in milliseconds from session start
     */
    sendAudioChunk(data: ArrayBuffer | Uint8Array, timestampMs: number): void;
    /**
     * Stop the current transcription session.
     *
     * @param reason - Optional reason for stopping
     */
    stopSession(reason?: string): Promise<void>;
    /**
     * Close the WebSocket connection.
     *
     * @param code - Close code (default 1000 = normal)
     * @param reason - Close reason
     */
    close(code?: number, reason?: string): void;
    /**
     * Check if the client is connected.
     */
    get isConnected(): boolean;
    /**
     * Check if a session is currently active.
     */
    get hasActiveSession(): boolean;
    /**
     * Build the WebSocket URL with auth query params.
     *
     * Note: This is a dev-only workaround. Browsers don't allow custom headers
     * on WebSocket connections. In production, we would use a background script
     * with webRequest API to inject headers.
     */
    private buildWebSocketUrl;
    /**
     * Send a message envelope to the server.
     */
    private sendEnvelope;
    /**
     * Handle incoming WebSocket message.
     */
    private handleMessage;
    /**
     * Dispatch envelope to appropriate handler.
     */
    private dispatchEnvelope;
    /**
     * Handle session_started message.
     */
    private handleSessionStarted;
    /**
     * Handle transcript message.
     */
    private handleTranscript;
    /**
     * Handle session_ended message.
     */
    private handleSessionEnded;
    /**
     * Handle error message.
     */
    private handleError;
    /**
     * Handle WebSocket close event.
     */
    private handleClose;
    /**
     * Emit an event to the callback.
     */
    private emit;
    /**
     * Convert Uint8Array to base64 string.
     */
    private arrayBufferToBase64;
}
//# sourceMappingURL=realtimeClient.d.ts.map