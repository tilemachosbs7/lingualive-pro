/**
 * =============================================================================
 * SETTINGS.TS — Extension Settings Storage
 * =============================================================================
 *
 * Κεντρική διαχείριση ρυθμίσεων για το LinguaLive Pro extension.
 *
 * Περιλαμβάνει:
 * - Backend URL configuration
 * - Dev token storage
 * - Debug headers fallback (user_id, plan)
 * - Quality mode preference
 *
 * ΣΗΜΕΙΩΣΕΙΣ:
 * - Για M2, το auth είναι dev-only (dev tokens ή debug headers).
 * - Σε production θα αντικατασταθεί από πραγματικό auth layer.
 * =============================================================================
 */
/**
 * Default settings for new installations.
 */
export const DEFAULT_SETTINGS = {
    backendBaseUrl: "http://localhost:8000",
    devToken: null,
    devTokenExpiresAt: null,
    debugUserId: null,
    debugPlan: null,
    preferredQualityMode: "FAST",
    sourceLanguage: "en",
    targetLanguage: "el",
};
/**
 * Storage key for persisting settings.
 */
const STORAGE_KEY = "lingualive_settings";
/**
 * Load settings from browser storage.
 * Returns default settings if none are stored.
 */
export async function loadSettings() {
    // For now, use a simple in-memory fallback
    // In real extension, this would use chrome.storage.local
    if (typeof chrome !== "undefined" && chrome.storage?.local) {
        return new Promise((resolve) => {
            chrome.storage.local.get([STORAGE_KEY], (result) => {
                const stored = result[STORAGE_KEY];
                if (stored) {
                    resolve({ ...DEFAULT_SETTINGS, ...stored });
                }
                else {
                    resolve({ ...DEFAULT_SETTINGS });
                }
            });
        });
    }
    // Fallback for non-extension contexts (testing, etc.)
    return { ...DEFAULT_SETTINGS };
}
/**
 * Save settings to browser storage.
 */
export async function saveSettings(settings) {
    if (typeof chrome !== "undefined" && chrome.storage?.local) {
        const current = await loadSettings();
        const updated = { ...current, ...settings };
        return new Promise((resolve) => {
            chrome.storage.local.set({ [STORAGE_KEY]: updated }, () => {
                resolve();
            });
        });
    }
    // Fallback: no-op for non-extension contexts
}
/**
 * Get the WebSocket URL for real-time captions.
 * Converts HTTP base URL to WS protocol.
 */
export function getWebSocketUrl(settings) {
    const base = settings.backendBaseUrl;
    // Convert http(s):// to ws(s)://
    const wsBase = base.replace(/^http/, "ws");
    return `${wsBase}/ws/captions`;
}
/**
 * Get the HTTP API base URL.
 */
export function getApiBaseUrl(settings) {
    return settings.backendBaseUrl;
}
//# sourceMappingURL=settings.js.map