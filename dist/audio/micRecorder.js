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
// =============================================================================
// MIC RECORDER CLASS
// =============================================================================
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
export class MicRecorder {
    options;
    mediaStream = null;
    mediaRecorder = null;
    startedAtMs = 0;
    isRecording = false;
    constructor(options) {
        this.options = options;
    }
    // ===========================================================================
    // PUBLIC API
    // ===========================================================================
    /**
     * Start recording from the microphone.
     *
     * Requests microphone permission and begins capturing audio.
     * Chunks are delivered via the onChunk callback.
     *
     * @throws Error if getUserMedia fails (permission denied, etc.)
     */
    async start() {
        if (this.isRecording) {
            throw new Error("Already recording");
        }
        // Request microphone access
        // Αυτό θα εμφανίσει το browser permission dialog
        try {
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: true,
                video: false,
            });
        }
        catch (err) {
            const error = err instanceof Error ? err : new Error("Microphone access denied");
            throw error;
        }
        // Determine MIME type
        const preferredMimeType = this.options.mimeType ?? "audio/webm;codecs=opus";
        const mimeType = MediaRecorder.isTypeSupported(preferredMimeType)
            ? preferredMimeType
            : undefined; // Let browser choose default
        // Create MediaRecorder
        try {
            this.mediaRecorder = new MediaRecorder(this.mediaStream, {
                mimeType,
            });
        }
        catch (err) {
            // Fallback: create without mimeType option
            this.mediaRecorder = new MediaRecorder(this.mediaStream);
        }
        // Handle data available events
        this.mediaRecorder.addEventListener("dataavailable", (event) => {
            void this.handleDataAvailable(event);
        });
        // Handle errors
        this.mediaRecorder.addEventListener("error", (event) => {
            const error = new Error(`MediaRecorder error: ${event.message ?? "unknown"}`);
            this.options.onError?.(error);
        });
        // Record start time
        this.startedAtMs = performance.now();
        this.isRecording = true;
        // Start recording with timeslice
        // Το timeslice ορίζει πόσο συχνά θα λαμβάνουμε chunks
        const timesliceMs = this.options.timesliceMs ?? 250;
        this.mediaRecorder.start(timesliceMs);
        console.log(`[MicRecorder] Started recording with mimeType: ${this.mediaRecorder.mimeType}`);
    }
    /**
     * Stop recording.
     *
     * Stops the MediaRecorder but keeps the MediaStream alive.
     * Call dispose() to fully release resources.
     */
    stop() {
        if (!this.isRecording || !this.mediaRecorder) {
            return;
        }
        if (this.mediaRecorder.state !== "inactive") {
            try {
                this.mediaRecorder.stop();
            }
            catch {
                // Ignore errors if already stopped
            }
        }
        this.isRecording = false;
        console.log("[MicRecorder] Stopped recording");
    }
    /**
     * Dispose of all resources.
     *
     * Stops recording and releases the microphone.
     * Call this when you're done with the recorder.
     */
    dispose() {
        this.stop();
        // Stop all tracks on the MediaStream
        // Αυτό απελευθερώνει το μικρόφωνο
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach((track) => {
                track.stop();
            });
            this.mediaStream = null;
        }
        this.mediaRecorder = null;
        console.log("[MicRecorder] Disposed");
    }
    /**
     * Check if currently recording.
     */
    get recording() {
        return this.isRecording;
    }
    // ===========================================================================
    // PRIVATE METHODS
    // ===========================================================================
    /**
     * Handle dataavailable event from MediaRecorder.
     */
    async handleDataAvailable(event) {
        // Skip empty chunks
        if (!event.data || event.data.size === 0) {
            return;
        }
        try {
            // Convert Blob to ArrayBuffer
            const arrayBuffer = await event.data.arrayBuffer();
            // Calculate timestamp from recording start
            const nowMs = performance.now();
            const timestampMs = Math.round(nowMs - this.startedAtMs);
            // Deliver to callback
            this.options.onChunk(arrayBuffer, timestampMs);
        }
        catch (err) {
            const error = err instanceof Error ? err : new Error("Failed to read audio chunk");
            this.options.onError?.(error);
        }
    }
}
//# sourceMappingURL=micRecorder.js.map