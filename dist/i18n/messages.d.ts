/**
 * =============================================================================
 * I18N/MESSAGES.TS — Internationalization Layer for LinguaLive Pro Extension
 * =============================================================================
 *
 * Purpose:
 * --------
 * Provides a simple but solid i18n foundation for the extension UI.
 * Centralizes all user-facing strings with support for Greek (primary) and English.
 *
 * Usage:
 * ------
 * import { t } from '../i18n/messages';
 * const text = t('hud.status.connecting');
 *
 * Notes:
 * ------
 * - Greek ('el') is the primary/default language.
 * - Locale is auto-detected from navigator.language, falling back to 'el'.
 * - No runtime locale settings yet; this is a foundation layer.
 * =============================================================================
 */
/**
 * Supported locales.
 * 'el' = Greek (primary), 'en' = English
 */
export type Locale = 'el' | 'en';
/**
 * Message keys for type-safe i18n.
 * Using a string union so TypeScript can validate key usage.
 */
export type MessageKey = 'hud.title' | 'hud.description' | 'hud.status.idle' | 'hud.status.connecting' | 'hud.status.connected' | 'hud.status.sessionStarted' | 'hud.status.live' | 'hud.status.stopping' | 'hud.status.disconnected' | 'hud.status.error' | 'hud.status.noAuth' | 'hud.status.authError' | 'hud.status.usageLimit' | 'hud.status.planLimit' | 'hud.status.usingFastMode' | 'hud.status.loadingLimits' | 'hud.status.loadingUsage' | 'hud.auth.required' | 'hud.auth.tokenExpired' | 'hud.auth.openSettings' | 'hud.limits.reached' | 'hud.limits.sessionReached' | 'hud.limits.rateLimited' | 'hud.limits.upgrade' | 'hud.limits.startButtonTitle' | 'hud.diagnostics.link' | 'hud.smartWarning.notAvailable' | 'hud.smartWarning.upgradeHint' | 'hud.transcript.placeholder' | 'hud.upgrade.link' | 'hud.usage.label' | 'hud.usage.minutes' | 'hud.usage.sessions' | 'hud.usage.sessionSingular' | 'options.diagnostics.title' | 'options.diagnostics.description' | 'options.diagnostics.runButton' | 'options.diagnostics.runningButton' | 'options.diagnostics.copyButton' | 'options.diagnostics.copyFirst' | 'options.diagnostics.copySuccess' | 'options.diagnostics.copyFailed' | 'options.diagnostics.clipboardNotSupported' | 'options.diagnostics.checkError' | 'options.diagnostics.reportTitle' | 'options.diagnostics.results' | 'options.diagnostics.result.environment' | 'options.diagnostics.result.backend' | 'options.diagnostics.result.auth' | 'options.diagnostics.result.microphone' | 'options.diagnostics.env.ok' | 'options.diagnostics.env.chromeApiMissing' | 'options.diagnostics.env.missingFeatures' | 'options.diagnostics.backend.ok' | 'options.diagnostics.backend.unreachable' | 'options.diagnostics.backend.reachableAuthSeparate' | 'options.diagnostics.backend.skipped' | 'options.diagnostics.auth.notConnected' | 'options.diagnostics.auth.connected' | 'options.diagnostics.auth.invalid' | 'options.diagnostics.auth.unreachable' | 'options.diagnostics.mic.granted' | 'options.diagnostics.mic.denied' | 'options.diagnostics.mic.prompt' | 'options.diagnostics.mic.skipped' | 'options.diagnostics.mic.skippedBrowser' | 'options.diagnostics.result.backendStatus' | 'options.diagnostics.backendStatus.ok' | 'options.diagnostics.backendStatus.unreachable' | 'options.diagnostics.backendStatus.error' | 'options.diagnostics.result.realtimeWebSocket' | 'options.diagnostics.realtimeWs.ok' | 'options.diagnostics.realtimeWs.skippedNoAuth' | 'options.diagnostics.realtimeWs.skippedNoBackend' | 'options.diagnostics.realtimeWs.error' | 'options.connection.testing' | 'options.connection.button' | 'options.connection.noAuth' | 'options.connection.invalidToken' | 'options.connection.serverNotFound' | 'options.connection.success' | 'options.backend.generateDevToken.button' | 'options.backend.generateDevToken.generating' | 'options.backend.generateDevToken.success' | 'options.backend.generateDevToken.errorMissingUser' | 'options.backend.generateDevToken.errorMissingBackend' | 'options.backend.generateDevToken.errorHttp' | 'options.backend.generateDevToken.errorNetwork' | 'options.backend.devTokenExpiry.label' | 'options.backend.devTokenExpiry.none' | 'options.backend.revokeDevToken.button' | 'options.backend.revokeDevToken.revoking' | 'options.backend.revokeDevToken.success' | 'options.backend.revokeDevToken.errorNoToken' | 'options.backend.revokeDevToken.errorHttp' | 'options.backend.revokeDevToken.errorNetwork' | 'options.summary.title' | 'options.summary.plan' | 'options.summary.usage' | 'options.summary.loading' | 'options.summary.loadError' | 'options.summary.noAuth' | 'options.summary.minutes' | 'options.summary.sessions' | 'options.summary.sessionSingular' | 'options.summary.perSession' | 'options.summary.maxPerSession' | 'options.language.label' | 'options.language.auto' | 'options.language.el' | 'options.language.en' | 'options.buttons.testConnection' | 'options.buttons.copyDiagnostics' | 'options.shortcuts.sectionTitle' | 'options.shortcuts.howToStart' | 'options.shortcuts.howToStart.steps' | 'options.shortcuts.keyboardHeading' | 'options.shortcuts.keyboardToggle' | 'options.shortcuts.keyboardOpenOptions' | 'options.shortcuts.contextMenuHeading' | 'options.shortcuts.contextMenuDescription' | 'options.shortcuts.chromeShortcutsNote' | 'options.help.sectionTitle' | 'options.help.whenToUseDiagnostics' | 'options.help.stepsHeading' | 'options.help.stepCheckBackend' | 'options.help.stepCheckMic' | 'options.help.stepCopyReport' | 'options.help.privacyNote' | 'options.error.generic';
/**
 * Set the current locale.
 * Use this to programmatically change the language.
 *
 * @param locale - The locale to set
 */
export declare function setLocale(locale: Locale): void;
/**
 * Get the current locale.
 *
 * @returns The current active locale
 */
export declare function getLocale(): Locale;
/**
 * Get a translated message for the current locale.
 *
 * Lookup order:
 * 1. Current locale's message
 * 2. Greek (primary) fallback
 * 3. Message key itself (if no translation found)
 *
 * @param key - The message key to look up
 * @returns The translated message string
 */
export declare function t(key: MessageKey): string;
//# sourceMappingURL=messages.d.ts.map