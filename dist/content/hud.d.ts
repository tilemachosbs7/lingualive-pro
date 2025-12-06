/**
 * =============================================================================
 * HUD.TS — Content Script HUD for LinguaLive Pro
 * =============================================================================
 *
 * Τι κάνει αυτό το module;
 * ------------------------
 * Injects ένα μικρό overlay HUD στις σελίδες YouTube που:
 * - Εμφανίζει captions σε real-time
 * - Διαχειρίζεται WebSocket connection μέσω RealtimeClient
 * - Καταγράφει ήχο από το μικρόφωνο (dev mode)
 * - Παρέχει Start/Stop controls
 *
 * ΧΡΗΣΗ:
 * ------
 * Φορτώνεται αυτόματα ως content script σε YouTube pages.
 * Δεν χρειάζεται manual initialization.
 *
 * ΣΗΜΕΙΩΣΕΙΣ:
 * -----------
 * - Χρησιμοποιεί mic capture (dev-only) για audio input.
 * - Χρησιμοποιεί dev auth (debug headers ή dev token).
 * - Σε production θα χρησιμοποιηθεί tabCapture για video audio.
 * =============================================================================
 */
export {};
//# sourceMappingURL=hud.d.ts.map