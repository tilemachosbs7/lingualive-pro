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

import {
  RealtimeClient,
  type RealtimeClientEvent,
  type QualityMode,
} from "../api/realtimeClient";
import { loadSettings, type LinguaLiveSettings } from "../storage/settings";
import { MicRecorder } from "../audio/micRecorder";
import {
  fetchUserBootstrap,
  fetchUserLimitsSummary,
  fetchUserUsageSummary,
  type UserBootstrapResponse,
  type UserLimitsSummary,
  type PublicUserUsageSummary,
} from "../api/backendClient";
import { t, setLocale } from "../i18n/messages";

// =============================================================================
// STATE
// =============================================================================

let client: RealtimeClient | null = null;
let micRecorder: MicRecorder | null = null;
let isRunning = false;
let lastPartialText = "";
let transcriptTextSoFar = "";

// Audio source: "mic" (default) or "tab" (tab audio capture)
let audioSource: "mic" | "tab" = "mic";
let usingTabCapture = false;

// Cached user plan/limits data
let cachedBootstrap: UserBootstrapResponse | null = null;
let cachedLimits: UserLimitsSummary | null = null;
let planDataLoading = false;
let planDataError: string | null = null;

// Cached user usage data
let cachedUsageSummary: PublicUserUsageSummary | null = null;
let usageLoading = false;
let usageError: string | null = null;

// DOM references (set after injection)
let container: HTMLElement | null = null;
let statusEl: HTMLDivElement | null = null;
let sessionConfigEl: HTMLDivElement | null = null;
let planBadgeEl: HTMLSpanElement | null = null;
let limitsInfoEl: HTMLDivElement | null = null;
let usageInfoEl: HTMLDivElement | null = null;
let startBtn: HTMLButtonElement | null = null;
let stopBtn: HTMLButtonElement | null = null;
let errorEl: HTMLDivElement | null = null;
let transcriptEl: HTMLDivElement | null = null;
let closeBtn: HTMLButtonElement | null = null;
let upgradeNudgeEl: HTMLDivElement | null = null;
let smartWarningEl: HTMLDivElement | null = null;
let authRequiredEl: HTMLDivElement | null = null;
let openSettingsBtn: HTMLButtonElement | null = null;
let limitsExceededEl: HTMLDivElement | null = null;
let limitsExceededUpgradeBtn: HTMLAnchorElement | null = null;

// Focus management for accessibility
let previouslyFocusedElement: Element | null = null;

// =============================================================================
// UI HELPERS
// =============================================================================

/**
 * Update the status text.
 */
function setStatus(text: string): void {
  if (statusEl) {
    statusEl.textContent = text;
  }
}

/**
 * Show or hide error message.
 * Optionally shows a "Διάγνωση προβλήματος" link for connection/technical errors.
 *
 * @param message - Error message to show, or null to hide
 * @param showDiagnosticsLink - If true, show a link to open Options page diagnostics
 */
function setError(message: string | null, showDiagnosticsLink = false): void {
  if (!errorEl) return;

  if (message) {
    // Build error content with optional diagnostics link
    if (showDiagnosticsLink) {
      errorEl.innerHTML = `
        ${escapeHtml(message)}
        <a href="#" id="llp-diag-link" style="color:#ff8080;text-decoration:underline;margin-left:8px;">${t('hud.diagnostics.link')}</a>
      `;
      // Wire up diagnostics link
      const diagLink = errorEl.querySelector<HTMLAnchorElement>("#llp-diag-link");
      if (diagLink) {
        diagLink.addEventListener("click", (e) => {
          e.preventDefault();
          chrome.runtime.openOptionsPage();
        });
      }
    } else {
      errorEl.textContent = message;
    }
    errorEl.style.display = "block";
  } else {
    errorEl.innerHTML = "";
    errorEl.style.display = "none";
  }
}

/**
 * Escape HTML to prevent XSS when using innerHTML.
 */
function escapeHtml(text: string): string {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Show the "auth required" message with CTA to open settings.
 * Used when user has no valid auth configured or token is invalid/expired.
 *
 * Auth-related errors include:
 * - No auth credentials configured (no token, no debug user)
 * - HTTP 401/403 from backend
 * - "invalid_token" or "expired_token" error codes from WebSocket
 */
function showAuthRequired(message?: string): void {
  if (!authRequiredEl) return;

  const defaultMessage = t('hud.auth.required');
  const textEl = authRequiredEl.querySelector<HTMLSpanElement>("#llp-auth-message");
  if (textEl) {
    textEl.textContent = message ?? defaultMessage;
  }
  authRequiredEl.style.display = "block";

  // Hide the regular error element to avoid confusion
  if (errorEl) {
    errorEl.style.display = "none";
  }
}

/**
 * Hide the "auth required" message.
 */
function hideAuthRequired(): void {
  if (!authRequiredEl) return;
  authRequiredEl.style.display = "none";
}

/**
 * Check if an error is auth-related.
 * Auth errors should show the "connect account" UX, not generic error.
 *
 * @param error - The error object or message
 * @param statusCode - Optional HTTP status code
 * @returns true if this is an auth-related error
 */
function isAuthError(error: unknown, statusCode?: number): boolean {
  // HTTP 401 (Unauthorized) or 403 (Forbidden)
  if (statusCode === 401 || statusCode === 403) {
    return true;
  }

  // Check error message for auth-related keywords
  const message = error instanceof Error ? error.message.toLowerCase() : String(error).toLowerCase();
  const authKeywords = [
    "unauthorized",
    "unauthenticated",
    "invalid token",
    "expired token",
    "invalid_token",
    "expired_token",
    "no auth",
    "auth required",
    "authentication",
    "not authenticated",
  ];

  return authKeywords.some(keyword => message.includes(keyword));
}

/**
 * Render the transcript (final + partial).
 */
function renderTranscript(): void {
  if (!transcriptEl) return;

  transcriptEl.innerHTML = "";

  // Final transcript block
  if (transcriptTextSoFar) {
    const finalBlock = document.createElement("div");
    finalBlock.textContent = transcriptTextSoFar;
    transcriptEl.appendChild(finalBlock);
  }

  // Partial transcript block (dimmed)
  if (lastPartialText) {
    const partialBlock = document.createElement("div");
    partialBlock.style.opacity = "0.6";
    partialBlock.style.fontStyle = "italic";
    partialBlock.textContent = lastPartialText;
    transcriptEl.appendChild(partialBlock);
  }

  // Placeholder if nothing yet
  if (!transcriptTextSoFar && !lastPartialText) {
    const placeholder = document.createElement("div");
    placeholder.style.color = "#888";
    placeholder.textContent = t('hud.transcript.placeholder');
    transcriptEl.appendChild(placeholder);
  }

  // Auto-scroll to bottom
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

/**
 * Append final transcript text.
 */
function appendFinalTranscript(text: string): void {
  if (transcriptTextSoFar) {
    transcriptTextSoFar += " ";
  }
  transcriptTextSoFar += text;
}

/**
 * Reset transcript state.
 */
function resetTranscript(): void {
  transcriptTextSoFar = "";
  lastPartialText = "";
  renderTranscript();
}

// =============================================================================
// PLAN BADGE & LIMITS DISPLAY
// =============================================================================

/**
 * Render the plan badge with styling based on tier.
 */
function renderPlanBadge(): void {
  if (!planBadgeEl) return;

  if (planDataLoading) {
    planBadgeEl.textContent = "…";
    planBadgeEl.style.background = "rgba(255,255,255,0.1)";
    planBadgeEl.style.color = "#888";
    return;
  }

  if (planDataError || !cachedBootstrap) {
    planBadgeEl.textContent = "?";
    planBadgeEl.style.background = "rgba(255,80,80,0.2)";
    planBadgeEl.style.color = "#ff8080";
    planBadgeEl.title = planDataError ?? "Failed to load plan info";
    return;
  }

  const tier = cachedBootstrap.plan.plan_tier;
  planBadgeEl.textContent = tier;
  planBadgeEl.title = `Plan: ${cachedBootstrap.plan.display_name}`;

  // Style based on tier
  switch (tier) {
    case "PRO":
      planBadgeEl.style.background = "rgba(76, 175, 80, 0.2)";
      planBadgeEl.style.color = "#4caf50";
      break;
    case "ENTERPRISE":
      planBadgeEl.style.background = "rgba(156, 39, 176, 0.2)";
      planBadgeEl.style.color = "#ba68c8";
      break;
    default: // FREE
      planBadgeEl.style.background = "rgba(255, 193, 7, 0.2)";
      planBadgeEl.style.color = "#ffc107";
      break;
  }
}

/**
 * Render the limits summary line.
 */
function renderLimitsInfo(): void {
  if (!limitsInfoEl) return;

  if (planDataLoading) {
    limitsInfoEl.textContent = t('hud.status.loadingLimits');
    return;
  }

  if (planDataError || !cachedLimits) {
    limitsInfoEl.textContent = planDataError ?? "Failed to load limits";
    limitsInfoEl.style.color = "#ff8080";
    return;
  }

  // Build limits summary
  const parts: string[] = [];

  // Audio limit
  const audioLimit = cachedLimits.per_session_audio_limit_minutes;
  if (audioLimit === null) {
    parts.push("Unlimited audio");
  } else {
    parts.push(`${audioLimit} min/session`);
  }

  // Quality modes
  const modes = cachedLimits.allowed_quality_modes;
  if (modes.length === 1) {
    parts.push(`${modes[0]} only`);
  } else if (modes.length > 0) {
    parts.push(modes.join("/"));
  }

  limitsInfoEl.textContent = parts.join(" · ");
  limitsInfoEl.style.color = "#aaa";
}

/**
 * Fetch plan and limits data from backend.
 */
async function fetchPlanData(): Promise<void> {
  planDataLoading = true;
  planDataError = null;
  renderPlanBadge();
  renderLimitsInfo();

  try {
    const settings = await loadSettings();

    // Check if auth is configured
    if (!settings.devToken && !settings.debugUserId) {
      throw new Error("No auth configured");
    }

    // Fetch both in parallel
    const [bootstrap, limits] = await Promise.all([
      fetchUserBootstrap(settings),
      fetchUserLimitsSummary(settings),
    ]);

    cachedBootstrap = bootstrap;
    cachedLimits = limits;
    planDataError = null;
  } catch (err) {
    console.warn("[LinguaLive] Failed to fetch plan data", err);
    planDataError = err instanceof Error ? err.message : "Failed to load";
    cachedBootstrap = null;
    cachedLimits = null;
  } finally {
    planDataLoading = false;
    renderPlanBadge();
    renderLimitsInfo();
    renderUpgradeNudge();
  }
}

// =============================================================================
// USAGE SUMMARY DISPLAY
// =============================================================================

/**
 * Render the usage summary line.
 */
function renderUsageInfo(): void {
  if (!usageInfoEl) return;

  if (usageLoading) {
    usageInfoEl.textContent = t('hud.status.loadingUsage');
    usageInfoEl.style.color = "#888";
    usageInfoEl.title = "";
    return;
  }

  if (usageError || !cachedUsageSummary) {
    usageInfoEl.textContent = "Usage: ?";
    usageInfoEl.style.color = "#ff8080";
    usageInfoEl.title = usageError ?? "Failed to load usage";
    return;
  }

  // Parse audio_minutes_total (comes as string from backend Decimal)
  const audioMinutes = parseFloat(cachedUsageSummary.audio_minutes_total);
  const displayMinutes = isNaN(audioMinutes) ? 0 : Math.round(audioMinutes);
  const sessions = cachedUsageSummary.total_sessions;

  // Build usage line
  let usageText = `Usage: ${displayMinutes} min`;
  if (sessions > 0) {
    usageText += ` · ${sessions} session${sessions !== 1 ? "s" : ""}`;
  }

  usageInfoEl.textContent = usageText;
  usageInfoEl.title = `Total: ${audioMinutes.toFixed(2)} min audio, ${cachedUsageSummary.transcript_chars_total.toLocaleString()} chars transcribed`;

  // Color based on usage (if we have a per-session limit from cached limits)
  // For now, just use neutral color since usage is cumulative across all sessions
  usageInfoEl.style.color = "#aaa";
}

/**
 * Fetch usage data from backend.
 */
async function fetchUsageData(): Promise<void> {
  usageLoading = true;
  usageError = null;
  renderUsageInfo();

  try {
    const settings = await loadSettings();

    // Check if auth is configured
    if (!settings.devToken && !settings.debugUserId) {
      throw new Error("No auth configured");
    }

    const usage = await fetchUserUsageSummary(settings);
    cachedUsageSummary = usage;
    usageError = null;
  } catch (err) {
    console.warn("[LinguaLive] Failed to fetch usage data", err);
    usageError = err instanceof Error ? err.message : "Failed to load";
    cachedUsageSummary = null;
  } finally {
    usageLoading = false;
    renderUsageInfo();
  }
}

// =============================================================================
// UPGRADE NUDGE & SMART WARNING
// =============================================================================

/**
 * Render the upgrade nudge for FREE users.
 * Shows a subtle "Upgrade" link near the plan/usage info.
 */
function renderUpgradeNudge(): void {
  if (!upgradeNudgeEl) return;

  // Only show for FREE tier users
  const isFreeUser = cachedBootstrap?.plan.plan_tier === "FREE";

  if (!isFreeUser || planDataLoading || planDataError) {
    upgradeNudgeEl.style.display = "none";
    return;
  }

  upgradeNudgeEl.style.display = "inline";
}

/**
 * Show a warning when SMART mode is not available for the user's plan.
 * Called when `quality_not_allowed_for_plan` event is received.
 */
function showSmartModeWarning(): void {
  if (!smartWarningEl) return;

  // Check if SMART is actually disallowed
  const allowedModes = cachedLimits?.allowed_quality_modes ?? [];
  const smartAllowed = allowedModes.includes("SMART");

  if (smartAllowed) {
    smartWarningEl.style.display = "none";
    return;
  }

  // Show warning with upgrade hint for FREE users
  const isFreeUser = cachedBootstrap?.plan.plan_tier === "FREE";
  const upgradeHint = isFreeUser ? " Upgrade for SMART mode." : "";
  smartWarningEl.textContent = `⚠️ SMART mode not available.${upgradeHint}`;
  smartWarningEl.style.display = "block";

  // Auto-hide after 8 seconds
  setTimeout(() => {
    if (smartWarningEl) {
      smartWarningEl.style.display = "none";
    }
  }, 8000);
}

/**
 * Hide the SMART mode warning.
 */
function hideSmartModeWarning(): void {
  if (!smartWarningEl) {
    return;
  }
  smartWarningEl.style.display = "none";
}

// =============================================================================
// LIMITS EXCEEDED UI
// =============================================================================

/**
 * Check if this is a limits/rate-limit related error or reason.
 * Used to detect when to show the "limits exceeded" UX.
 *
 * Maps backend semantics:
 * - ErrorCode.RATE_LIMITED (from error events)
 * - reason: "limit_reached" (from session_ended events)
 * - LimitViolation codes like "audio_minutes", "translated_chars"
 *
 * @param value - The error code, reason string, or error message
 * @returns true if this indicates a limit violation
 */
function isLimitsError(value: unknown): boolean {
  if (!value) return false;

  const str = String(value).toLowerCase();

  // Backend error codes and reasons for limits
  const limitKeywords = [
    "rate_limited",
    "rate-limited",
    "limit_reached",
    "limit-reached",
    "limits_exceeded",
    "limits-exceeded",
    "audio_minutes",
    "translated_chars",
    "quota",
    "exceeded",
    "όριο",  // Greek "limit"
  ];

  return limitKeywords.some(keyword => str.includes(keyword));
}

/**
 * Show the "limits exceeded" message with upgrade CTA.
 * Stops the session gracefully and shows a clear Greek message.
 *
 * @param message - Optional custom message (Greek). Falls back to default.
 */
function showLimitsExceeded(message?: string): void {
  if (!limitsExceededEl) return;

  const defaultMessage = t('hud.limits.reached');
  const messageEl = limitsExceededEl.querySelector<HTMLSpanElement>("#llp-limits-message");

  if (messageEl) {
    messageEl.textContent = message ?? defaultMessage;
  }

  // Show the limits exceeded element
  limitsExceededEl.style.display = "block";

  // Hide any regular error to avoid confusion
  if (errorEl) {
    errorEl.style.display = "none";
  }

  // Make sure upgrade button is visible for FREE users
  if (limitsExceededUpgradeBtn) {
    const isFreeUser = cachedBootstrap?.plan.plan_tier === "FREE";
    limitsExceededUpgradeBtn.style.display = isFreeUser ? "inline-block" : "none";
  }

  // Update start button to show limit status
  if (startBtn) {
    startBtn.disabled = false; // Allow retry, but message will show again
    startBtn.title = t('hud.limits.startButtonTitle');
  }
}

/**
 * Hide the "limits exceeded" message.
 */
function hideLimitsExceeded(): void {
  if (!limitsExceededEl) return;
  limitsExceededEl.style.display = "none";

  // Reset start button title
  if (startBtn) {
    startBtn.title = "";
  }
}

// =============================================================================
// TAB AUDIO CHUNK HANDLER
// =============================================================================

/**
 * Handle incoming tab audio chunks from background service worker.
 * Προωθεί τα audio chunks από το tab capture στον RealtimeClient.
 */
function handleTabChunkMessage(message: unknown): void {
  const m = message as { type?: string; buffer?: ArrayBuffer; timestampMs?: number };
  if (m.type !== "LLP_TAB_AUDIO_CHUNK") return;
  if (!client || !m.buffer || typeof m.timestampMs !== "number") return;

  try {
    client.sendAudioChunk(m.buffer, m.timestampMs);
  } catch (err) {
    console.warn("[LinguaLive] Failed to send tab audio chunk", err);
  }
}

/**
 * Toggle captions: if running, stop; if not running, start.
 * Used by keyboard shortcut (Alt+Shift+C).
 */
async function toggleCaptions(): Promise<void> {
  if (isRunning) {
    await handleStop();
  } else {
    await handleStart();
  }
}

/**
 * Runtime message listener for tab audio chunks and keyboard shortcuts.
 * Named function για να μπορεί να αφαιρεθεί με removeListener.
 */
function runtimeMessageListener(message: unknown): void {
  // Check for toggle captions command from background script
  if (
    typeof message === "object" &&
    message !== null &&
    (message as { type?: unknown }).type === "LLP_TOGGLE_CAPTIONS"
  ) {
    void toggleCaptions();
    return;
  }

  // Otherwise, handle tab audio chunks
  handleTabChunkMessage(message);
}

// =============================================================================
// EVENT HANDLER
// =============================================================================

/**
 * Handle events from RealtimeClient.
 */
function handleEvent(evt: RealtimeClientEvent): void {
  switch (evt.type) {
    case "connecting":
      setStatus(t('hud.status.connecting'));
      break;

    case "open":
      setStatus(t('hud.status.connected'));
      break;

    case "session_started":
      setStatus(t('hud.status.sessionStarted'));
      break;

    case "partial_transcript":
      lastPartialText = evt.text;
      renderTranscript();
      break;

    case "final_transcript":
      appendFinalTranscript(evt.text);
      lastPartialText = "";
      renderTranscript();
      break;

    case "session_ended":
      // Check if session ended due to plan limits being reached
      if (isLimitsError(evt.reason)) {
        // Show Greek message for limit reached
        showLimitsExceeded(t('hud.limits.sessionReached'));
        setStatus(t('hud.status.planLimit'));
      } else {
        setStatus(`Session ended: ${evt.reason}`);
      }
      isRunning = false;
      if (startBtn) startBtn.disabled = false;
      if (stopBtn) stopBtn.disabled = true;
      break;

    case "quality_not_allowed_for_plan":
      // Show the dedicated SMART warning UI instead of error
      showSmartModeWarning();
      setStatus(t('hud.status.usingFastMode'));
      break;

    case "error":
      console.error("[LinguaLive] Realtime error", evt.error, evt.errorCode);
      // Check if this is an auth-related error
      if (isAuthError(evt.error, evt.errorCode === "unauthorized" ? 401 : undefined)) {
        showAuthRequired(t('hud.auth.tokenExpired'));
        setStatus(t('hud.status.authError'));
      } else if (isLimitsError(evt.errorCode) || isLimitsError(evt.error.message)) {
        // RATE_LIMITED or other limit-related error
        showLimitsExceeded(t('hud.limits.rateLimited'));
        setStatus(t('hud.status.usageLimit'));
      } else {
        // Connection/server error - show diagnostics link
        setError(evt.error.message, true);
        setStatus(t('hud.status.error'));
      }
      isRunning = false;
      if (startBtn) startBtn.disabled = false;
      if (stopBtn) stopBtn.disabled = true;
      break;

    case "closed":
      setStatus(t('hud.status.disconnected'));
      isRunning = false;
      if (startBtn) startBtn.disabled = false;
      if (stopBtn) stopBtn.disabled = true;
      break;
  }
}

// =============================================================================
// START/STOP HANDLERS
// =============================================================================

/**
 * Start a new transcription session.
 */
async function handleStart(): Promise<void> {
  if (isRunning) return;

  // Reset UI
  setError(null);
  hideAuthRequired();
  hideSmartModeWarning();
  hideLimitsExceeded();
  resetTranscript();
  setStatus(t('hud.status.connecting'));

  if (startBtn) startBtn.disabled = true;
  if (stopBtn) stopBtn.disabled = false;

  // Load settings
  let settings: LinguaLiveSettings;
  try {
    settings = await loadSettings();
    // Apply language preference to i18n system
    if (settings.language === 'el' || settings.language === 'en') {
      setLocale(settings.language);
    }
  } catch (err) {
    console.error("[LinguaLive] Failed to load settings", err);
    setError("No settings found. Configure LinguaLive in Options.", true);
    setStatus(t('hud.status.idle'));
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
    return;
  }

  // Check auth is configured
  if (!settings.devToken && !settings.debugUserId) {
    showAuthRequired();
    setStatus(t('hud.status.noAuth'));
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
    return;
  }

  // Create client
  // Χρησιμοποιούμε τα defaultSourceLang/defaultTargetLang/defaultQualityMode από τα settings.
  // Αν δεν υπάρχουν, κάνουμε fallback σε "en" / "FAST".
  const qualityMode: QualityMode =
    (settings.defaultQualityMode as QualityMode | undefined) ?? settings.preferredQualityMode ?? "FAST";
  const sourceLang = settings.defaultSourceLang?.trim() || settings.sourceLanguage || "en";
  const targetLang = settings.defaultTargetLang?.trim() || settings.targetLanguage || "en";

  // Update session config summary in HUD
  if (sessionConfigEl) {
    sessionConfigEl.textContent = `${sourceLang} → ${targetLang} · ${qualityMode}`;
  }

  client = new RealtimeClient({
    settings,
    qualityMode,
    onEvent: handleEvent,
  });

  // Connect + start session
  try {
    await client.connect();
    await client.startSession({
      sourceLang,
      targetLang,
    });
  } catch (err) {
    console.error("[LinguaLive] Failed to start session", err);

    // Check for auth errors
    const statusCode = (err as { status?: number }).status;
    if (isAuthError(err, statusCode)) {
      showAuthRequired(t('hud.auth.tokenExpired'));
      setStatus(t('hud.status.authError'));
    } else {
      // Connection/session errors - show diagnostics link
      setError(err instanceof Error ? err.message : "Failed to start session", true);
      setStatus(t('hud.status.idle'));
    }

    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
    client = null;
    isRunning = false;
    return;
  }

  // Start audio capture based on selected source
  // Ανάλογα με την επιλογή του χρήστη, ξεκινάμε mic ή tab capture.
  if (audioSource === "mic") {
    // Microphone capture path
    // Δημιουργούμε τον MicRecorder και στέλνουμε chunks στον client
    micRecorder = new MicRecorder({
      onChunk: (buffer, timestampMs) => {
        if (!client) return;
        try {
          client.sendAudioChunk(buffer, timestampMs);
        } catch (err) {
          console.warn("[LinguaLive] Failed to send audio chunk", err);
        }
      },
      onError: (error) => {
        console.error("[LinguaLive] Mic error", error);
        setError("Microphone error: " + error.message, true);
      },
    });

    try {
      await micRecorder.start();
      usingTabCapture = false;
      isRunning = true;
      setStatus(t('hud.status.live'));
    } catch (err) {
      console.error("[LinguaLive] Failed to start microphone", err);
      setError("Microphone permission denied or unavailable.", true);
      // Clean up WS session since we have no audio
      try {
        await client.stopSession("mic_failed");
        client.close();
      } catch {
        // Ignore cleanup errors
      }
      micRecorder = null;
      client = null;
      isRunning = false;
      setStatus(t('hud.status.idle'));
      if (startBtn) startBtn.disabled = false;
      if (stopBtn) stopBtn.disabled = true;
      return;
    }
  } else {
    // Tab audio capture path
    // Η συλλογή audio γίνεται στο background service worker
    try {
      const response = await chrome.runtime.sendMessage({ type: "LLP_START_TAB_CAPTURE" });
      if (!response || !response.ok) {
        throw new Error(response?.error ?? "Failed to start tab capture");
      }
      usingTabCapture = true;
      isRunning = true;
      setStatus(t('hud.status.live'));
    } catch (err) {
      console.error("[LinguaLive] Failed to start tab capture", err);
      setError(err instanceof Error ? err.message : "Failed to start tab capture.", true);
      // Clean up WS session
      try {
        await client.stopSession("tab_capture_failed");
        client.close();
      } catch {
        // Ignore cleanup errors
      }
      client = null;
      isRunning = false;
      setStatus(t('hud.status.idle'));
      if (startBtn) startBtn.disabled = false;
      if (stopBtn) stopBtn.disabled = true;
      return;
    }
  }
}

/**
 * Stop the current session.
 */
async function handleStop(): Promise<void> {
  if (!client || !isRunning) return;

  setStatus(t('hud.status.stopping'));
  if (stopBtn) stopBtn.disabled = true;

  // Stop microphone if using mic capture
  if (micRecorder) {
    micRecorder.stop();
    micRecorder.dispose();
    micRecorder = null;
  }

  // Stop tab capture if using tab audio
  if (usingTabCapture) {
    try {
      await chrome.runtime.sendMessage({ type: "LLP_STOP_TAB_CAPTURE" });
    } catch {
      // Best-effort cleanup
    }
    usingTabCapture = false;
  }

  // Then stop WS session
  try {
    await client.stopSession("user_stop");
    client.close();
  } catch (err) {
    console.warn("[LinguaLive] Error while stopping session", err);
  }

  client = null;
  isRunning = false;
  setStatus(t('hud.status.idle'));
  if (startBtn) startBtn.disabled = false;
}

// =============================================================================
// ACCESSIBILITY HELPERS
// =============================================================================

/**
 * Selector for focusable elements within the HUD.
 * Standard approach for focus trap implementation.
 */
const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'textarea:not([disabled])',
  'select:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(', ');

/**
 * Get all focusable elements within the HUD container.
 */
function getFocusableElements(): HTMLElement[] {
  if (!container) return [];
  return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR));
}

/**
 * Handle focus trap within the HUD.
 * Ensures Tab and Shift+Tab cycle within the dialog.
 */
function handleFocusTrap(event: KeyboardEvent): void {
  if (event.key !== 'Tab' || !container) return;

  const focusableElements = getFocusableElements();
  if (focusableElements.length === 0) return;

  const firstElement = focusableElements[0];
  const lastElement = focusableElements[focusableElements.length - 1];

  if (event.shiftKey) {
    // Shift+Tab: if on first element, wrap to last
    if (document.activeElement === firstElement) {
      event.preventDefault();
      lastElement.focus();
    }
  } else {
    // Tab: if on last element, wrap to first
    if (document.activeElement === lastElement) {
      event.preventDefault();
      firstElement.focus();
    }
  }
}

/**
 * Move focus to the first focusable element in the HUD (close button).
 */
function focusFirstElement(): void {
  // Prefer close button as initial focus target
  if (closeBtn) {
    closeBtn.focus();
    return;
  }
  // Fallback to first focusable element
  const focusableElements = getFocusableElements();
  if (focusableElements.length > 0) {
    focusableElements[0].focus();
  }
}

/**
 * Restore focus to the previously focused element.
 */
function restoreFocus(): void {
  if (
    previouslyFocusedElement &&
    previouslyFocusedElement instanceof HTMLElement &&
    document.body.contains(previouslyFocusedElement)
  ) {
    previouslyFocusedElement.focus();
  } else {
    // Fallback: blur current focus (returns to body)
    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
    }
  }
  previouslyFocusedElement = null;
}

// =============================================================================
// CLOSE HUD HELPER
// =============================================================================

/**
 * Keydown handler for Escape key and focus trap.
 * Δημιουργείται ως named function ώστε να μπορεί να αφαιρεθεί σωστά με removeEventListener.
 */
function handleKeydown(event: KeyboardEvent): void {
  // Escape: close the HUD
  if (event.key === "Escape") {
    event.stopPropagation();
    event.preventDefault();
    void closeHud();
    return;
  }

  // Tab/Shift+Tab: handle focus trap
  if (event.key === "Tab") {
    handleFocusTrap(event);
  }
}

/**
 * Close the HUD and cleanup.
 * Κεντρική function που χρησιμοποιείται από:
 * - Close button click
 * - Escape key handler
 * - beforeunload (partial cleanup)
 */
async function closeHud(): Promise<void> {
  // Remove keydown listener first
  window.removeEventListener("keydown", handleKeydown);

  // Restore focus to previously focused element (a11y)
  restoreFocus();

  // Remove runtime message listener
  chrome.runtime.onMessage.removeListener(runtimeMessageListener);

  // Stop microphone if running
  if (micRecorder) {
    micRecorder.stop();
    micRecorder.dispose();
    micRecorder = null;
  }

  // Stop tab capture if active
  if (usingTabCapture) {
    try {
      await chrome.runtime.sendMessage({ type: "LLP_STOP_TAB_CAPTURE" });
    } catch {
      // Best-effort cleanup
    }
    usingTabCapture = false;
  }

  // Stop session if running
  if (client && isRunning) {
    try {
      await client.stopSession("user_close");
      client.close();
    } catch {
      // Ignore errors during cleanup
    }
    client = null;
    isRunning = false;
  }

  // Remove HUD from DOM
  if (container && container.parentNode) {
    container.parentNode.removeChild(container);
  }
  container = null;
}

// =============================================================================
// HUD INJECTION
// =============================================================================

/**
 * Create and inject the HUD into the page.
 */
function injectHud(): void {
  // Don't inject twice
  if (document.getElementById("lingualive-hud")) {
    return;
  }

  // Create container
  // Μη-modal overlay - δεν "κλειδώνει" τον χρήστη, απλά εμφανίζει captions.
  // Save the currently focused element for later restoration (a11y)
  previouslyFocusedElement = document.activeElement;

  container = document.createElement("section");
  container.id = "lingualive-hud";
  container.setAttribute("role", "dialog");
  container.setAttribute("aria-modal", "true");
  container.setAttribute("aria-labelledby", "llp-hud-title");
  container.setAttribute("aria-describedby", "llp-hud-description");
  // Make container focusable for focus management
  container.setAttribute("tabindex", "-1");

  // Styles
  Object.assign(container.style, {
    position: "fixed",
    bottom: "16px",
    right: "16px",
    zIndex: "2147483647",
    background: "rgba(0, 0, 0, 0.9)",
    color: "#fff",
    padding: "12px",
    borderRadius: "10px",
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
    fontSize: "13px",
    maxWidth: "380px",
    minWidth: "280px",
    boxShadow: "0 4px 20px rgba(0, 0, 0, 0.5)",
    border: "1px solid rgba(255, 255, 255, 0.1)",
  });

  // Inner HTML
  // Note: Some strings use i18n t() function for localization
  container.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
      <div>
        <div style="display:flex;align-items:center;gap:8px;">
          <strong id="llp-hud-title" style="font-size:14px;">${t('hud.title')}</strong>
          <span id="llp-plan-badge" style="font-size:10px;padding:2px 6px;border-radius:4px;font-weight:600;">…</span>
          <span id="llp-upgrade-nudge" role="link" tabindex="0" style="display:none;font-size:9px;color:#4fc3f7;cursor:pointer;text-decoration:underline;" title="Get SMART mode and higher limits" aria-label="Upgrade to Pro plan">${t('hud.upgrade.link')}</span>
        </div>
        <p id="llp-hud-description" style="position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);border:0;">${t('hud.description')}</p>
        <div id="llp-limits-info" style="font-size:10px;color:#aaa;margin-top:2px;">${t('hud.status.loadingLimits')}</div>
        <div id="llp-usage-info" style="font-size:10px;color:#888;margin-top:2px;">${t('hud.status.loadingUsage')}</div>
        <div id="llp-session-config" style="font-size:11px;color:#aaa;margin-top:2px;"></div>
        <div id="llp-status" style="display:inline-block;font-size:10px;color:#fff;margin-top:4px;padding:2px 8px;border-radius:10px;background:rgba(255,255,255,0.15);font-weight:500;">${t('hud.status.idle')}</div>
      </div>
      <button id="llp-close-btn" type="button" aria-label="Close LinguaLive captions" style="background:none;border:none;color:#888;cursor:pointer;font-size:18px;padding:4px;line-height:1;">✕</button>
    </div>
    <div id="llp-smart-warning" style="display:none;color:#ffd54f;background:rgba(255,213,79,0.1);padding:6px 8px;border-radius:4px;font-size:11px;margin-bottom:8px;">
      ${t('hud.smartWarning.notAvailable')}
    </div>
    <div id="llp-auth-required" role="alert" style="display:none;color:#4fc3f7;background:rgba(79,195,247,0.1);padding:8px;border-radius:4px;font-size:11px;margin-bottom:8px;">
      <span id="llp-auth-message">${t('hud.auth.required')}</span>
      <button id="llp-open-settings-btn" type="button" style="display:block;margin-top:6px;padding:4px 10px;border-radius:4px;border:1px solid #4fc3f7;background:transparent;color:#4fc3f7;cursor:pointer;font-size:11px;font-weight:500;" aria-label="${t('hud.auth.openSettings')}">⚙️ ${t('hud.auth.openSettings')}</button>
    </div>
    <div id="llp-limits-exceeded" role="alert" aria-live="polite" style="display:none;color:#ff9800;background:rgba(255,152,0,0.15);padding:8px;border-radius:4px;font-size:11px;margin-bottom:8px;border:1px solid rgba(255,152,0,0.3);">
      <span id="llp-limits-message">${t('hud.limits.reached')}</span>
      <a id="llp-limits-upgrade-btn" href="https://lingualive.pro/pricing" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin-top:6px;padding:5px 12px;border-radius:4px;background:#ff9800;color:#fff;text-decoration:none;font-size:11px;font-weight:600;" aria-label="${t('hud.limits.upgrade')}">⬆ ${t('hud.limits.upgrade')}</a>
    </div>
    <fieldset id="llp-audio-source" style="display:flex;gap:12px;margin-bottom:10px;font-size:12px;color:#ccc;border:none;padding:0;margin:0 0 10px 0;">
      <legend style="position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);border:0;">Audio source</legend>
      <label style="cursor:pointer;display:flex;align-items:center;gap:4px;">
        <input type="radio" name="llp-audio-source" value="mic" checked style="margin:0;" aria-describedby="llp-audio-source-legend" />
        Mic
      </label>
      <label style="cursor:pointer;display:flex;align-items:center;gap:4px;">
        <input type="radio" name="llp-audio-source" value="tab" style="margin:0;" aria-describedby="llp-audio-source-legend" />
        Tab audio
      </label>
    </fieldset>
    <div style="display:flex;gap:8px;margin-bottom:10px;">
      <button id="llp-start-btn" type="button" aria-label="Start live captions" style="flex:1;padding:6px 12px;border-radius:6px;border:none;background:#4caf50;color:#fff;cursor:pointer;font-weight:500;font-size:13px;">▶ Start</button>
      <button id="llp-stop-btn" type="button" aria-label="Stop live captions" style="flex:1;padding:6px 12px;border-radius:6px;border:none;background:#444;color:#aaa;cursor:not-allowed;font-weight:500;font-size:13px;" disabled>■ Stop</button>
    </div>
    <div id="llp-error" role="alert" aria-live="assertive" style="display:none;color:#ff8080;background:rgba(255,0,0,0.1);padding:6px 8px;border-radius:4px;font-size:11px;margin-bottom:8px;"></div>
    <div id="llp-transcript" aria-label="Live captions" style="max-height:200px;overflow-y:auto;font-size:13px;line-height:1.5;padding:8px;background:rgba(255,255,255,0.05);border-radius:6px;">
      <div style="color:#888;">${t('hud.transcript.placeholder')}</div>
    </div>
  `;

  // Append to body
  document.body.appendChild(container);

  // Grab references
  statusEl = container.querySelector<HTMLDivElement>("#llp-status");
  sessionConfigEl = container.querySelector<HTMLDivElement>("#llp-session-config");
  planBadgeEl = container.querySelector<HTMLSpanElement>("#llp-plan-badge");
  limitsInfoEl = container.querySelector<HTMLDivElement>("#llp-limits-info");
  usageInfoEl = container.querySelector<HTMLDivElement>("#llp-usage-info");
  upgradeNudgeEl = container.querySelector<HTMLDivElement>("#llp-upgrade-nudge");
  smartWarningEl = container.querySelector<HTMLDivElement>("#llp-smart-warning");
  startBtn = container.querySelector<HTMLButtonElement>("#llp-start-btn");
  stopBtn = container.querySelector<HTMLButtonElement>("#llp-stop-btn");
  errorEl = container.querySelector<HTMLDivElement>("#llp-error");
  transcriptEl = container.querySelector<HTMLDivElement>("#llp-transcript");
  closeBtn = container.querySelector<HTMLButtonElement>("#llp-close-btn");
  authRequiredEl = container.querySelector<HTMLDivElement>("#llp-auth-required");
  openSettingsBtn = container.querySelector<HTMLButtonElement>("#llp-open-settings-btn");
  limitsExceededEl = container.querySelector<HTMLDivElement>("#llp-limits-exceeded");
  limitsExceededUpgradeBtn = container.querySelector<HTMLAnchorElement>("#llp-limits-upgrade-btn");

  // A11y: status text → screen readers ακούν ήπια τις αλλαγές
  if (statusEl) {
    statusEl.setAttribute("role", "status");
    statusEl.setAttribute("aria-live", "polite");
    statusEl.setAttribute("aria-atomic", "true");
  }

  // A11y: transcript → "log" + aria-relevant=additions
  // Βοηθά τα screen readers να διαβάζουν νέα κομμάτια χωρίς να επαναλαμβάνουν όλο το κείμενο.
  if (transcriptEl) {
    transcriptEl.setAttribute("role", "log");
    transcriptEl.setAttribute("aria-live", "polite");
    transcriptEl.setAttribute("aria-atomic", "false");
    transcriptEl.setAttribute("aria-relevant", "additions");
  }

  // Audio source selector
  // Επιτρέπει εναλλαγή μεταξύ mic και tab audio capture.
  const audioSourceInputs = container.querySelectorAll<HTMLInputElement>(
    'input[name="llp-audio-source"]'
  );
  audioSourceInputs.forEach((input) => {
    input.addEventListener("change", () => {
      if (input.checked && (input.value === "mic" || input.value === "tab")) {
        audioSource = input.value;
      }
    });
  });

  // Runtime message listener for tab audio chunks
  // Ακούει chunks από το background service worker.
  chrome.runtime.onMessage.addListener(runtimeMessageListener);

  // Wire up event listeners
  if (startBtn) {
    startBtn.addEventListener("click", () => {
      void handleStart();
    });
  }

  if (stopBtn) {
    stopBtn.addEventListener("click", () => {
      void handleStop();
    });
  }

  if (closeBtn) {
    closeBtn.addEventListener("click", () => {
      void closeHud();
    });
  }

  // Upgrade nudge click/keyboard handler - open pricing page
  if (upgradeNudgeEl) {
    const openPricingPage = (): void => {
      window.open("https://lingualive.pro/pricing", "_blank");
    };
    upgradeNudgeEl.addEventListener("click", openPricingPage);
    upgradeNudgeEl.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        openPricingPage();
      }
    });
  }

  // Open Settings button click/keyboard handler - opens extension options page
  if (openSettingsBtn) {
    const openOptionsPage = (): void => {
      // Use Chrome runtime API to open the extension's options page
      chrome.runtime.openOptionsPage();
    };
    openSettingsBtn.addEventListener("click", openOptionsPage);
    openSettingsBtn.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        openOptionsPage();
      }
    });
  }

  // Keyboard: Escape closes HUD and Tab traps focus
  window.addEventListener("keydown", handleKeydown);

  // Focus management: move focus to close button (a11y)
  // Small delay to ensure DOM is ready
  requestAnimationFrame(() => {
    focusFirstElement();
  });

  // Cleanup on page unload
  window.addEventListener("beforeunload", () => {
    // Remove keydown listener
    window.removeEventListener("keydown", handleKeydown);
    // Remove runtime message listener
    chrome.runtime.onMessage.removeListener(runtimeMessageListener);
    // Stop microphone
    if (micRecorder) {
      try {
        micRecorder.stop();
        micRecorder.dispose();
      } catch {
        // Ignore
      }
    }
    // Stop tab capture if active
    if (usingTabCapture) {
      try {
        chrome.runtime.sendMessage({ type: "LLP_STOP_TAB_CAPTURE" });
      } catch {
        // Best-effort
      }
    }
    // Close WebSocket
    if (client && isRunning) {
      try {
        client.close(1000, "page_unload");
      } catch {
        // Ignore errors during cleanup
      }
    }
  });

  console.log("[LinguaLive] HUD injected");

  // Fetch plan data in background (non-blocking)
  // This populates the plan badge and limits info
  void fetchPlanData();

  // Fetch usage data in background (non-blocking)
  // This populates the usage summary line
  void fetchUsageData();
}

// =============================================================================
// INITIALIZATION
// =============================================================================

/**
 * Initialize the HUD content script.
 * Loads settings first to apply language preference before rendering.
 */
async function init(): Promise<void> {
  // Load settings to apply language preference before rendering HUD
  try {
    const settings = await loadSettings();
    if (settings.language === 'el' || settings.language === 'en') {
      setLocale(settings.language);
    }
  } catch {
    // Ignore errors, will use auto-detected locale
  }

  if (document.body) {
    injectHud();
  } else {
    window.addEventListener(
      "DOMContentLoaded",
      () => {
        injectHud();
      },
      { once: true }
    );
  }
}

// Run on script load
void init();
