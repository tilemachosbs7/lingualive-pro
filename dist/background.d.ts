/**
 * =============================================================================
 * BACKGROUND.TS — Background Service Worker for LinguaLive Pro
 * =============================================================================
 *
 * Τι κάνει αυτό το module;
 * ------------------------
 * Υλοποιεί tab audio capture χρησιμοποιώντας chrome.tabCapture API.
 * Αυτό είναι dev-only tab audio capture για το τρέχον active tab όπου είναι ο HUD.
 *
 * Ακούει μηνύματα από το HUD content script:
 * - LLP_START_TAB_CAPTURE: ξεκινά capture από το current tab
 * - LLP_STOP_TAB_CAPTURE: σταματά το capture
 *
 * Στέλνει audio chunks πίσω στο content script ως:
 * - LLP_TAB_AUDIO_CHUNK: ArrayBuffer + timestampMs
 *
 * ΣΗΜΕΙΩΣΕΙΣ:
 * -----------
 * - Χρησιμοποιεί MediaRecorder για 250ms slices.
 * - Απαιτεί "tabCapture" permission στο manifest.
 * - Σε production, μπορεί να χρειαστεί πιο robust error handling.
 * =============================================================================
 */
export type TabAudioChunkMessage = {
    type: "LLP_TAB_AUDIO_CHUNK";
    buffer: ArrayBuffer;
    timestampMs: number;
};
//# sourceMappingURL=background.d.ts.map