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
    defaultSourceLang?: string;
    defaultTargetLang?: string;
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
export declare const DEFAULT_SETTINGS: LinguaLiveSettings;
/**
 * Load settings from browser storage.
 * Returns default settings if none are stored.
 */
export declare function loadSettings(): Promise<LinguaLiveSettings>;
/**
 * Save settings to browser storage.
 */
export declare function saveSettings(settings: Partial<LinguaLiveSettings>): Promise<void>;
/**
 * Get the WebSocket URL for real-time captions.
 * Converts HTTP base URL to WS protocol.
 */
export declare function getWebSocketUrl(settings: LinguaLiveSettings): string;
/**
 * Get the HTTP API base URL.
 */
export declare function getApiBaseUrl(settings: LinguaLiveSettings): string;
//# sourceMappingURL=settings.d.ts.map