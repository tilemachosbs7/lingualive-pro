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
import { getWebSocketUrl, } from "../storage/settings";
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
    options;
    ws = null;
    sessionId = null;
    isSessionActive = false;
    sequenceNumber = 0;
    constructor(options) {
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
    async connect() {
        if (this.ws) {
            throw new Error("Already connected");
        }
        this.emit({ type: "connecting" });
        const wsUrl = this.buildWebSocketUrl();
        this.ws = new WebSocket(wsUrl);
        return new Promise((resolve, reject) => {
            const ws = this.ws;
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
    async startSession(params) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            throw new Error("WebSocket not connected");
        }
        if (this.isSessionActive) {
            throw new Error("Session already active");
        }
        // Generate a new session ID
        this.sessionId = crypto.randomUUID();
        this.sequenceNumber = 0;
        const payload = {
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
    sendAudioChunk(data, timestampMs) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            throw new Error("WebSocket not connected");
        }
        if (!this.isSessionActive || !this.sessionId) {
            throw new Error("No active session");
        }
        // Convert to base64
        const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
        const chunkBase64 = this.arrayBufferToBase64(bytes);
        const payload = {
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
    async stopSession(reason) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            return; // Already disconnected
        }
        if (!this.isSessionActive || !this.sessionId) {
            return; // No active session
        }
        const payload = {
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
    close(code = 1000, reason = "Client closed") {
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
    get isConnected() {
        return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
    }
    /**
     * Check if a session is currently active.
     */
    get hasActiveSession() {
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
    buildWebSocketUrl() {
        const { settings } = this.options;
        const baseUrl = getWebSocketUrl(settings);
        const url = new URL(baseUrl);
        // Add auth as query params (dev workaround)
        // Priority: dev token > debug headers
        if (settings.devToken) {
            url.searchParams.set("dev_token", settings.devToken);
        }
        else if (settings.debugUserId) {
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
    sendEnvelope(envelope) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            throw new Error("WebSocket not connected");
        }
        this.ws.send(JSON.stringify(envelope));
    }
    /**
     * Handle incoming WebSocket message.
     */
    handleMessage(event) {
        let envelope;
        try {
            envelope = JSON.parse(event.data);
        }
        catch {
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
    dispatchEnvelope(envelope) {
        switch (envelope.type) {
            case "session_started":
                this.handleSessionStarted(envelope.payload);
                break;
            case "transcript":
                this.handleTranscript(envelope.payload);
                break;
            case "session_ended":
                this.handleSessionEnded(envelope.payload);
                break;
            case "error":
                this.handleError(envelope.payload);
                break;
            default:
                // Ignore unknown message types
                console.warn(`[RealtimeClient] Unknown message type: ${envelope.type}`);
        }
    }
    /**
     * Handle session_started message.
     */
    handleSessionStarted(payload) {
        this.isSessionActive = true;
        this.emit({ type: "session_started", payload });
    }
    /**
     * Handle transcript message.
     */
    handleTranscript(payload) {
        if (payload.is_final) {
            this.emit({
                type: "final_transcript",
                text: payload.text,
                translatedText: payload.translated_text,
                timestampMs: payload.timestamp_ms ?? null,
            });
        }
        else {
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
    handleSessionEnded(payload) {
        this.isSessionActive = false;
        this.emit({ type: "session_ended", reason: payload.reason });
    }
    /**
     * Handle error message.
     */
    handleError(payload) {
        // Check for quality_not_allowed_for_plan error (backend sends uppercase)
        if (payload.code === "QUALITY_NOT_ALLOWED_FOR_PLAN" ||
            payload.code === "quality_not_allowed_for_plan") {
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
    handleClose(event) {
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
    emit(event) {
        try {
            this.options.onEvent(event);
        }
        catch (err) {
            console.error("[RealtimeClient] Error in event handler:", err);
        }
    }
    /**
     * Convert Uint8Array to base64 string.
     */
    arrayBufferToBase64(bytes) {
        let binary = "";
        for (let i = 0; i < bytes.length; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }
}
//# sourceMappingURL=realtimeClient.js.map