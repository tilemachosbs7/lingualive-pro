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
 * Plan tiers supported by LinguaLive Pro.
 */
export type PlanTier = "FREE" | "PRO" | "ENTERPRISE";

/**
 * Quality modes for real-time transcription.
 */
export type QualityMode = "FAST" | "SMART";

/**
 * Settings stored by the extension.
 */
export interface LinguaLiveSettings {
  /**
   * Base URL for the backend API (HTTP).
   * Example: "http://localhost:8000"
   */
  backendBaseUrl: string;

  /**
   * Dev token for authentication.
   * If present, takes priority over debug headers.
   */
  devToken: string | null;

  /**
   * Dev token expiration timestamp (ISO 8601 string).
   * Stored alongside devToken for display purposes.
   */
  devTokenExpiresAt: string | null;

  /**
   * Debug user ID (fallback when no dev token).
   * Used for development/testing.
   */
  debugUserId: string | null;

  /**
   * Debug plan tier (fallback when no dev token).
   * Defaults to FREE if not set.
   */
  debugPlan: PlanTier | null;

  /**
   * Preferred quality mode for transcription.
   */
  preferredQualityMode: QualityMode;

  /**
   * Source language for transcription.
   */
  sourceLanguage: string;

  /**
   * Target language for translation.
   */
  targetLanguage: string;

  // Προεπιλεγμένη γλώσσα πηγής για το realtime session (π.χ. "en").
  defaultSourceLang?: string;

  // Προεπιλεγμένη γλώσσα στόχου (π.χ. "el").
  defaultTargetLang?: string;

  // Προεπιλεγμένο quality mode για το WS pipeline ("FAST" ή "SMART").
  defaultQualityMode?: "FAST" | "SMART";

  /**
   * UI language preference.
   * - 'auto': Detect from navigator.language (default behavior)
   * - 'el': Greek
   * - 'en': English
   */
  language?: "auto" | "el" | "en";
}

/**
 * Default settings for new installations.
 */
export const DEFAULT_SETTINGS: LinguaLiveSettings = {
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
export async function loadSettings(): Promise<LinguaLiveSettings> {
  // For now, use a simple in-memory fallback
  // In real extension, this would use chrome.storage.local
  if (typeof chrome !== "undefined" && chrome.storage?.local) {
    return new Promise((resolve) => {
      chrome.storage.local.get([STORAGE_KEY], (result) => {
        const stored = result[STORAGE_KEY];
        if (stored) {
          resolve({ ...DEFAULT_SETTINGS, ...stored });
        } else {
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
export async function saveSettings(
  settings: Partial<LinguaLiveSettings>
): Promise<void> {
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
export function getWebSocketUrl(settings: LinguaLiveSettings): string {
  const base = settings.backendBaseUrl;
  // Convert http(s):// to ws(s)://
  const wsBase = base.replace(/^http/, "ws");
  return `${wsBase}/ws/captions`;
}

/**
 * Get the HTTP API base URL.
 */
export function getApiBaseUrl(settings: LinguaLiveSettings): string {
  return settings.backendBaseUrl;
}
