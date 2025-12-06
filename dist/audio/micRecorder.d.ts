/**
 * =============================================================================
 * MICRECORDER.TS — Microphone Audio Capture (Dev Module)
 * =============================================================================
 *
 * Τι κάνει αυτό το module;
 * ------------------------
 * Παρέχει ένα απλό wrapper γύρω από το MediaRecorder API για να:
 * - Ζητάει πρόσβαση στο μικρόφωνο
 * - Καταγράφει ήχο σε μικρά chunks
 * - Στέλνει τα chunks στον caller μέσω callback
 *
 * ΣΚΟΠΟΣ:
 * -------
 * Αυτό είναι dev-only module. Χρησιμοποιείται για testing του WebSocket
 * pipeline με πραγματικό ήχο από το μικρόφωνο.
 *
 * ΣΗΜΕΙΩΣΕΙΣ:
 * -----------
 * - Ο backend (mock ASR) δεν κάνει πραγματικό speech recognition.
 * - Τα audio chunks αντιμετωπίζονται ως opaque bytes από τον backend.
 * - Σε production θα χρησιμοποιηθεί tabCapture για capture video audio.
 *
 * ΧΡΗΣΗ:
 * ------
 *   const recorder = new MicRecorder({
 *     onChunk: (buffer, timestampMs) => {
 *       client.sendAudioChunk(buffer, timestampMs);
 *     },
 *     onError: (error) => console.error(error),
 *   });
 *   await recorder.start();
 *   // ... later
 *   recorder.stop();
 *   recorder.dispose();
 * =============================================================================
 */
/**
 * Options for MicRecorder.
 */
export interface MicRecorderOptions {
    /**
     * MIME type for the recording.
     * Default: "audio/webm;codecs=opus" (widely supported)
     */
    mimeType?: string;
    /**
     * Callback for each audio chunk.
     * @param data - Raw audio data as ArrayBuffer
     * @param timestampMs - Timestamp in milliseconds from recording start
     */
    onChunk: (data: ArrayBuffer, timestampMs: number) => void;
    /**
     * Optional callback for errors.
     */
    onError?: (error: Error) => void;
    /**
     * Timeslice in milliseconds for chunk frequency.
     * Default: 250ms (4 chunks per second)
     */
    timesliceMs?: number;
}
/**
 * Microphone audio recorder using MediaRecorder API.
 *
 * Lifecycle:
 * 1. Create instance with callbacks
 * 2. Call start() to request mic and begin recording
 * 3. Receive chunks via onChunk callback
 * 4. Call stop() to stop recording
 * 5. Call dispose() to release resources
 */
export declare class MicRecorder {
    private readonly options;
    private mediaStream;
    private mediaRecorder;
    private startedAtMs;
    private isRecording;
    constructor(options: MicRecorderOptions);
    /**
     * Start recording from the microphone.
     *
     * Requests microphone permission and begins capturing audio.
     * Chunks are delivered via the onChunk callback.
     *
     * @throws Error if getUserMedia fails (permission denied, etc.)
     */
    start(): Promise<void>;
    /**
     * Stop recording.
     *
     * Stops the MediaRecorder but keeps the MediaStream alive.
     * Call dispose() to fully release resources.
     */
    stop(): void;
    /**
     * Dispose of all resources.
     *
     * Stops recording and releases the microphone.
     * Call this when you're done with the recorder.
     */
    dispose(): void;
    /**
     * Check if currently recording.
     */
    get recording(): boolean;
    /**
     * Handle dataavailable event from MediaRecorder.
     */
    private handleDataAvailable;
}
//# sourceMappingURL=micRecorder.d.ts.map