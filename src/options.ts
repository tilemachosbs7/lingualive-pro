/**
 * =============================================================================
 * OPTIONS.TS — Options Page for LinguaLive Pro Extension
 * =============================================================================
 *
 * Τι κάνει αυτό το module;
 * ------------------------
 * Υλοποιεί την Options σελίδα του Chrome extension.
 * Φορτώνει/αποθηκεύει settings και καλεί τα backend endpoints:
 * - GET /user/bootstrap
 * - GET /user/limits-summary
 *
 * Εμφανίζει:
 * - Account info (user_id, auth source, plan tier)
 * - Plan limits (audio, translation per session)
 * - Usage & billing stats
 * - Preferences (default source/target language, default quality mode)
 *
 * ΣΗΜΕΙΩΣΕΙΣ:
 * -----------
 * - Μπορείς να αλλάξεις τις προτιμήσεις (languages/quality) και να τις αποθηκεύσεις.
 * - Τα errors εμφανίζονται στο UI (δεν καταπίνονται σιωπηλά).
 * =============================================================================
 */

import { loadSettings, saveSettings, type LinguaLiveSettings } from "./storage/settings";
import {
  fetchUserBootstrap,
  fetchUserLimitsSummary,
  fetchPublicStatus,
  requestDevToken,
  revokeDevToken,
  type UserBootstrapResponse,
  type UserLimitsSummary,
} from "./api/backendClient";
import { RealtimeClient } from "./api/realtimeClient";
import { t, setLocale } from "./i18n/messages";

// =============================================================================
// PREFERENCES ELEMENTS
// =============================================================================

let sourceLangInput: HTMLInputElement | null = null;
let targetLangInput: HTMLInputElement | null = null;
let qualitySelect: HTMLSelectElement | null = null;
let languageSelect: HTMLSelectElement | null = null;
let saveBtn: HTMLButtonElement | null = null;
let saveStatusEl: HTMLSpanElement | null = null;
let currentSettings: LinguaLiveSettings | null = null;

// Backend & Auth elements
let backendUrlInput: HTMLInputElement | null = null;
let devTokenInput: HTMLInputElement | null = null;
let debugUserIdInput: HTMLInputElement | null = null;
let debugPlanInput: HTMLInputElement | null = null;
let testConnectionBtn: HTMLButtonElement | null = null;
let connectionTestStatusEl: HTMLSpanElement | null = null;
let generateDevTokenBtn: HTMLButtonElement | null = null;
let generateDevTokenStatusEl: HTMLDivElement | null = null;
let revokeDevTokenBtn: HTMLButtonElement | null = null;
let devTokenExpiryEl: HTMLParagraphElement | null = null;

// Summary panel elements
let summaryPlanEl: HTMLSpanElement | null = null;
let summaryUsageEl: HTMLSpanElement | null = null;
let summaryUpgradeBtn: HTMLAnchorElement | null = null;
let summaryErrorEl: HTMLDivElement | null = null;
let summaryNoAuthEl: HTMLDivElement | null = null;
let summaryContentEl: HTMLDivElement | null = null;
let summaryLimitStatusEl: HTMLSpanElement | null = null;

// Limit status elements
let audioLimitStatusEl: HTMLSpanElement | null = null;
let translationLimitStatusEl: HTMLSpanElement | null = null;

// Diagnostics elements
let runDiagnosticsBtn: HTMLButtonElement | null = null;
let diagnosticsResultsEl: HTMLDivElement | null = null;
let diagnosticsListEl: HTMLDivElement | null = null;
let copyDiagnosticsBtn: HTMLButtonElement | null = null;
let copyStatusEl: HTMLSpanElement | null = null;

// =============================================================================
// LIMIT STATUS TYPES & HELPERS
// =============================================================================

/**
 * Limit status values.
 * Maps to backend PlanLimitStatus semantics:
 * - OK: within limits, no issues
 * - NEAR: approaching limit (>80% of limit used)
 * - EXCEEDED: over limit
 */
type LimitStatus = "OK" | "NEAR" | "EXCEEDED";

/**
 * Greek labels for limit status.
 * Used in both summary badge and limits section.
 * Reserved for future use when we have per-session usage data.
 */
const _LIMIT_STATUS_LABELS: Record<LimitStatus, string> = {
  OK: "Όριο: OK",
  NEAR: "Κοντά στο όριο",
  EXCEEDED: "Υπέρβαση ορίου",
};

/**
 * CSS class for limit status badge.
 * Reserved for future use when we have per-session usage data.
 */
const _LIMIT_STATUS_CLASSES: Record<LimitStatus, string> = {
  OK: "ok",
  NEAR: "near",
  EXCEEDED: "exceeded",
};

/**
 * Calculate limit status based on usage vs limit.
 * Uses 80% threshold for "near limit" warning.
 *
 * Note: Backend limits are per-session, but this function can be used
 * for any usage/limit comparison. Since we don't have per-session usage
 * here, this is mainly for display purposes.
 * Reserved for future use when we have per-session usage data.
 *
 * @param usage - Current usage value
 * @param limit - Limit value (null = unlimited)
 * @returns LimitStatus indicating OK, NEAR, or EXCEEDED
 */
function _calculateLimitStatus(usage: number, limit: number | null): LimitStatus {
  // No limit = always OK
  if (limit === null || limit === 0) {
    return "OK";
  }

  const percentage = (usage / limit) * 100;

  if (percentage >= 100) {
    return "EXCEEDED";
  } else if (percentage >= 80) {
    return "NEAR";
  }
  return "OK";
}

// =============================================================================
// DIAGNOSTICS TYPES & HELPERS
// =============================================================================

/**
 * Status of a diagnostic check.
 */
type DiagnosticStatus = "ok" | "error" | "info" | "skipped";

/**
 * Result of a single diagnostic check.
 */
interface DiagnosticResult {
  name: string;
  status: DiagnosticStatus;
  message: string;
  details?: string;
}

/**
 * Full diagnostics report.
 */
interface DiagnosticsReport {
  extensionVersion: string;
  browser: string;
  backendUrl: string;
  results: DiagnosticResult[];
  timestamp: string;
}

/**
 * Get a friendly browser name from user agent.
 */
function getBrowserName(): string {
  const ua = navigator.userAgent;
  if (ua.includes("Edg/")) {
    return "Microsoft Edge";
  } else if (ua.includes("Chrome/")) {
    return "Google Chrome";
  } else if (ua.includes("Firefox/")) {
    return "Mozilla Firefox";
  } else if (ua.includes("Safari/") && !ua.includes("Chrome")) {
    return "Safari";
  }
  return "Unknown Browser";
}

/**
 * Get extension version from manifest.
 */
function getExtensionVersion(): string {
  try {
    return chrome.runtime.getManifest().version;
  } catch {
    return "unknown";
  }
}

/**
 * Check if basic browser/extension environment is OK.
 */
function checkEnvironment(): DiagnosticResult {
  const hasChrome = typeof chrome !== "undefined" && chrome.runtime;
  const hasFetch = typeof fetch !== "undefined";
  const hasWebSocket = typeof WebSocket !== "undefined";

  if (!hasChrome) {
    return {
      name: t('options.diagnostics.result.environment'),
      status: "error",
      message: t('options.diagnostics.env.chromeApiMissing'),
    };
  }

  if (!hasFetch || !hasWebSocket) {
    return {
      name: t('options.diagnostics.result.environment'),
      status: "error",
      message: t('options.diagnostics.env.missingFeatures'),
    };
  }

  const version = getExtensionVersion();
  const browser = getBrowserName();

  return {
    name: t('options.diagnostics.result.environment'),
    status: "ok",
    message: t('options.diagnostics.env.ok'),
    details: `${browser}, Extension v${version}`,
  };
}

/**
 * Check if backend is reachable.
 * Uses fetchUserBootstrap if auth is configured, otherwise shows skipped.
 */
async function checkBackend(settings: LinguaLiveSettings): Promise<DiagnosticResult> {
  const hasAuth = Boolean(settings.devToken || settings.debugUserId);

  if (!hasAuth) {
    return {
      name: t('options.diagnostics.result.backend'),
      status: "skipped",
      message: t('options.diagnostics.backend.skipped'),
      details: `URL: ${settings.backendBaseUrl || "http://localhost:8000"}`,
    };
  }

  try {
    // Use the existing fetchUserBootstrap to test reachability
    await fetchUserBootstrap(settings, 5);
    return {
      name: t('options.diagnostics.result.backend'),
      status: "ok",
      message: t('options.diagnostics.backend.ok'),
      details: `URL: ${settings.backendBaseUrl || "http://localhost:8000"}`,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    const statusCode = (err as { status?: number }).status;

    // Network error (server down, wrong URL)
    if (message.includes("fetch") || message.includes("network") || message.includes("Failed to fetch")) {
      return {
        name: t('options.diagnostics.result.backend'),
        status: "error",
        message: t('options.diagnostics.backend.unreachable'),
        details: `URL: ${settings.backendBaseUrl || "http://localhost:8000"}`,
      };
    }

    // Auth error (server reachable but auth failed)
    if (statusCode === 401 || statusCode === 403) {
      return {
        name: t('options.diagnostics.result.backend'),
        status: "ok",
        message: t('options.diagnostics.backend.reachableAuthSeparate'),
        details: `URL: ${settings.backendBaseUrl || "http://localhost:8000"}`,
      };
    }

    return {
      name: t('options.diagnostics.result.backend'),
      status: "error",
      message: `${t('options.error.generic')}: ${message}`,
      details: `URL: ${settings.backendBaseUrl || "http://localhost:8000"}`,
    };
  }
}

/**
 * Check auth status.
 */
async function checkAuth(settings: LinguaLiveSettings): Promise<DiagnosticResult> {
  const hasAuth = Boolean(settings.devToken || settings.debugUserId);

  if (!hasAuth) {
    return {
      name: t('options.diagnostics.result.auth'),
      status: "skipped",
      message: t('options.diagnostics.auth.notConnected'),
    };
  }

  try {
    const bootstrap = await fetchUserBootstrap(settings, 5);
    return {
      name: t('options.diagnostics.result.auth'),
      status: "ok",
      message: `${t('options.diagnostics.auth.connected')} ${bootstrap.whoami.user_id} (${bootstrap.plan.plan_tier}).`,
    };
  } catch (err) {
    const statusCode = (err as { status?: number }).status;
    const message = err instanceof Error ? err.message : String(err);

    if (statusCode === 401 || statusCode === 403) {
      return {
        name: t('options.diagnostics.result.auth'),
        status: "error",
        message: t('options.diagnostics.auth.invalid'),
      };
    }

    if (message.includes("fetch") || message.includes("network") || message.includes("Failed to fetch")) {
      return {
        name: t('options.diagnostics.result.auth'),
        status: "error",
        message: t('options.diagnostics.auth.unreachable'),
      };
    }

    return {
      name: t('options.diagnostics.result.auth'),
      status: "error",
      message: `${t('options.error.generic')}: ${message}`,
    };
  }
}

/**
 * Check microphone permission status.
 */
async function checkMicrophone(): Promise<DiagnosticResult> {
  // Check if permissions API is available
  if (!navigator.permissions) {
    return {
      name: t('options.diagnostics.result.microphone'),
      status: "skipped",
      message: t('options.diagnostics.mic.skippedBrowser'),
    };
  }

  try {
    // Query microphone permission
    const result = await navigator.permissions.query({ name: "microphone" as PermissionName });

    switch (result.state) {
      case "granted":
        return {
          name: t('options.diagnostics.result.microphone'),
          status: "ok",
          message: t('options.diagnostics.mic.granted'),
        };
      case "denied":
        return {
          name: t('options.diagnostics.result.microphone'),
          status: "error",
          message: t('options.diagnostics.mic.denied'),
        };
      case "prompt":
        return {
          name: t('options.diagnostics.result.microphone'),
          status: "info",
          message: t('options.diagnostics.mic.prompt'),
        };
      default:
        return {
          name: t('options.diagnostics.result.microphone'),
          status: "info",
          message: `${t('options.summary.usage')}: ${result.state}`,
        };
    }
  } catch (err) {
    // Some browsers throw on microphone permission query
    console.warn("[options] Microphone permission query failed:", err);
    return {
      name: t('options.diagnostics.result.microphone'),
      status: "skipped",
      message: t('options.diagnostics.mic.skipped'),
    };
  }
}

/**
 * Check backend public status via GET /public/status.
 * This is a public endpoint that doesn't require authentication.
 */
async function checkBackendStatus(settings: LinguaLiveSettings): Promise<DiagnosticResult> {
  const backendUrl = settings.backendBaseUrl || "http://localhost:8000";

  try {
    const status = await fetchPublicStatus(backendUrl, 5);

    // Build details string: env=X, version=Y
    const details = `env=${status.environment}, v${status.version}`;

    return {
      name: t('options.diagnostics.result.backendStatus'),
      status: "ok",
      message: `${t('options.diagnostics.backendStatus.ok')} (${details})`,
      details: `${status.service}, latency: ${status.pipeline_latency_ms}ms`,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);

    // Network/fetch error - backend unreachable
    if (
      message.includes("fetch") ||
      message.includes("network") ||
      message.includes("Failed to fetch") ||
      message.includes("aborted")
    ) {
      return {
        name: t('options.diagnostics.result.backendStatus'),
        status: "error",
        message: t('options.diagnostics.backendStatus.unreachable'),
        details: `URL: ${backendUrl}`,
      };
    }

    // Other error
    return {
      name: t('options.diagnostics.result.backendStatus'),
      status: "error",
      message: `${t('options.diagnostics.backendStatus.error')}: ${message}`,
      details: `URL: ${backendUrl}`,
    };
  }
}

/**
 * Check WebSocket connectivity via /ws/captions.
 * This performs a minimal connect → close check, no full session.
 */
async function checkRealtimeWebSocket(settings: LinguaLiveSettings): Promise<DiagnosticResult> {
  const backendUrl = settings.backendBaseUrl || "";

  // Check if backend URL is configured
  if (!backendUrl) {
    return {
      name: t('options.diagnostics.result.realtimeWebSocket'),
      status: "skipped",
      message: t('options.diagnostics.realtimeWs.skippedNoBackend'),
    };
  }

  // Check if auth is configured
  const hasAuth = Boolean(settings.devToken || settings.debugUserId);
  if (!hasAuth) {
    return {
      name: t('options.diagnostics.result.realtimeWebSocket'),
      status: "skipped",
      message: t('options.diagnostics.realtimeWs.skippedNoAuth'),
    };
  }

  // Attempt a minimal WebSocket connection
  const client = new RealtimeClient({
    settings,
    qualityMode: "FAST",
    onEvent: () => {
      // No-op: we don't care about events for diagnostics
    },
  });

  try {
    // Create a timeout promise
    const timeoutMs = 5000;
    const connectWithTimeout = Promise.race([
      client.connect(),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error("WebSocket connection timeout")), timeoutMs)
      ),
    ]);

    await connectWithTimeout;

    return {
      name: t('options.diagnostics.result.realtimeWebSocket'),
      status: "ok",
      message: t('options.diagnostics.realtimeWs.ok'),
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);

    return {
      name: t('options.diagnostics.result.realtimeWebSocket'),
      status: "error",
      message: t('options.diagnostics.realtimeWs.error'),
      details: message,
    };
  } finally {
    // Best-effort cleanup to avoid leaving an open WS connection in diagnostics
    client.close();
  }
}

/**
 * Run all diagnostic checks and return a report.
 */
async function runDiagnostics(): Promise<DiagnosticsReport> {
  const settings = await loadSettings();
  const results: DiagnosticResult[] = [];

  // 1. Environment check (always runs)
  results.push(checkEnvironment());

  // 2. Backend status check (public endpoint, no auth required)
  results.push(await checkBackendStatus(settings));

  // 3. Backend check (with auth)
  results.push(await checkBackend(settings));

  // 4. Realtime WebSocket check
  results.push(await checkRealtimeWebSocket(settings));

  // 5. Auth check
  results.push(await checkAuth(settings));

  // 6. Microphone check
  results.push(await checkMicrophone());

  return {
    extensionVersion: getExtensionVersion(),
    browser: getBrowserName(),
    backendUrl: settings.backendBaseUrl || "http://localhost:8000",
    results,
    timestamp: new Date().toISOString(),
  };
}

/**
 * Render diagnostics results in the UI.
 */
function renderDiagnosticsResults(report: DiagnosticsReport): void {
  if (!diagnosticsListEl) return;

  const iconMap: Record<DiagnosticStatus, string> = {
    ok: "✅",
    error: "❌",
    info: "ℹ️",
    skipped: "⏭️",
  };

  let html = "";

  for (const result of report.results) {
    const icon = iconMap[result.status];
    const statusClass = result.status === "ok" ? "success"
      : result.status === "error" ? "error"
      : result.status === "info" ? "info"
      : "skipped";

    html += `
      <div class="diag-row">
        <span class="diag-icon">${icon}</span>
        <span class="diag-name">${result.name}</span>
        <span class="diag-result ${statusClass}">${result.message}${result.details ? ` <small style="color:#888;">(${result.details})</small>` : ""}</span>
      </div>
    `;
  }

  diagnosticsListEl.innerHTML = html;

  if (diagnosticsResultsEl) {
    diagnosticsResultsEl.style.display = "block";
  }
}

/**
 * Generate plain text report for clipboard.
 */
function generateDiagnosticsText(report: DiagnosticsReport): string {
  let text = t('options.diagnostics.reportTitle') + "\n";
  text += "================================\n\n";
  text += `Extension version: ${report.extensionVersion}\n`;
  text += `Browser: ${report.browser}\n`;
  text += `Backend URL: ${report.backendUrl}\n`;
  text += `Timestamp: ${report.timestamp}\n\n`;
  text += t('options.diagnostics.results') + ":\n";
  text += "-------------\n";

  for (const result of report.results) {
    const statusIcon = result.status === "ok" ? "[OK]"
      : result.status === "error" ? "[ERROR]"
      : result.status === "info" ? "[INFO]"
      : "[SKIP]";

    text += `${statusIcon} ${result.name}: ${result.message}`;
    if (result.details) {
      text += ` (${result.details})`;
    }
    text += "\n";
  }

  return text;
}

// Store last report for copying
let lastDiagnosticsReport: DiagnosticsReport | null = null;

/**
 * Handle running diagnostics.
 */
async function handleRunDiagnostics(): Promise<void> {
  if (!runDiagnosticsBtn || !diagnosticsListEl) return;

  runDiagnosticsBtn.disabled = true;
  runDiagnosticsBtn.textContent = t('options.diagnostics.runningButton');

  try {
    const report = await runDiagnostics();
    lastDiagnosticsReport = report;
    renderDiagnosticsResults(report);
  } catch (err) {
    console.error("[options] Diagnostics failed:", err);
    diagnosticsListEl.innerHTML = `<div class="diag-row"><span class="diag-icon">❌</span><span class="diag-result error">${t('options.diagnostics.checkError')}: ${err instanceof Error ? err.message : String(err)}</span></div>`;
    if (diagnosticsResultsEl) {
      diagnosticsResultsEl.style.display = "block";
    }
  } finally {
    runDiagnosticsBtn.disabled = false;
    runDiagnosticsBtn.textContent = t('options.diagnostics.runButton');
  }
}

/**
 * Handle copying diagnostics report to clipboard.
 */
async function handleCopyDiagnostics(): Promise<void> {
  if (!copyDiagnosticsBtn || !copyStatusEl) return;

  if (!lastDiagnosticsReport) {
    copyStatusEl.textContent = t('options.diagnostics.copyFirst');
    copyStatusEl.style.color = "#666";
    return;
  }

  try {
   const text = generateDiagnosticsText(lastDiagnosticsReport);

    // Check clipboard API availability
    if (!navigator.clipboard || !navigator.clipboard.writeText) {
      copyStatusEl.textContent = t('options.diagnostics.clipboardNotSupported');
      copyStatusEl.style.color = "#b00020";
      return;
    }

    await navigator.clipboard.writeText(text);
    copyStatusEl.textContent = t('options.diagnostics.copySuccess');
    copyStatusEl.style.color = "#2e7d32";

    // Clear message after 3 seconds
    setTimeout(() => {
      if (copyStatusEl) {
        copyStatusEl.textContent = "";
      }
    }, 3000);
  } catch (err) {
    console.error("[options] Failed to copy to clipboard:", err);
    copyStatusEl.textContent = t('options.diagnostics.copyFailed');
    copyStatusEl.style.color = "#b00020";
  }
}

// =============================================================================
// DOM HELPERS
// =============================================================================

/**
 * Get element by ID with type safety.
 * Throws if element is not found.
 */
function byId<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) {
    throw new Error(`Missing element with id="${id}" in options.html`);
  }
  return el as T;
}

/**
 * Set text content of an element by ID.
 * Safe helper that handles null checks.
 */
function setText(id: string, text: string): void {
  byId<HTMLElement>(id).textContent = text;
}

/**
 * Show or hide an element by ID.
 */
function setVisible(id: string, visible: boolean): void {
  byId<HTMLElement>(id).style.display = visible ? "block" : "none";
}

// =============================================================================
// UI UPDATE HELPERS
// =============================================================================

/**
 * Update the Account section with bootstrap data.
 */
function updateAccountSection(
  bootstrap: UserBootstrapResponse,
  limits: UserLimitsSummary
): void {
  setText("user-id", bootstrap.whoami.user_id);
  setText("auth-source", bootstrap.whoami.auth_source);

  // Update plan badge with styling
  const planBadge = byId<HTMLSpanElement>("plan-tier");
  planBadge.textContent = bootstrap.plan.plan_tier;
  planBadge.className = "badge";
  if (bootstrap.plan.plan_tier === "PRO") {
    planBadge.classList.add("pro");
  } else if (bootstrap.plan.plan_tier === "ENTERPRISE") {
    planBadge.classList.add("enterprise");
  }

  // Quality modes
  const modes = limits.allowed_quality_modes;
  setText(
    "allowed-quality-modes",
    modes.length > 0 ? modes.join(", ") : "None"
  );
}

/**
 * Update the Limits section with limits data.
 * For FREE plans, shows the per-session limits.
 * Note: These are per-session limits, not cumulative quotas.
 */
function updateLimitsSection(limits: UserLimitsSummary): void {
  // Audio limit
  const audioLimit = limits.per_session_audio_limit_minutes;
  setText(
    "audio-limit",
    audioLimit == null ? "Unlimited" : `${audioLimit.toFixed(1)} minutes`
  );

  // Audio limit status hint for FREE users
  if (audioLimitStatusEl) {
    if (limits.is_free_plan && audioLimit !== null) {
      audioLimitStatusEl.textContent = t('options.summary.perSession');
      audioLimitStatusEl.style.color = "#666";
    } else {
      audioLimitStatusEl.textContent = "";
    }
  }

  // Translation limit
  const translationLimit = limits.per_session_translation_chars_limit;
  setText(
    "translation-limit",
    translationLimit == null
      ? "Unlimited"
      : `${translationLimit.toLocaleString()} chars`
  );

  // Translation limit status hint for FREE users
  if (translationLimitStatusEl) {
    if (limits.is_free_plan && translationLimit !== null) {
      translationLimitStatusEl.textContent = t('options.summary.perSession');
      translationLimitStatusEl.style.color = "#666";
    } else {
      translationLimitStatusEl.textContent = "";
    }
  }
}

/**
 * Update the summary panel limit status badge.
 * Shows a compact indicator for FREE users about their plan limits.
 *
 * Since limits are per-session (not cumulative), we show:
 * - For FREE users: reminder about per-session limits
 * - For PRO/ENTERPRISE: no limits badge needed
 */
function updateSummaryLimitStatus(limits: UserLimitsSummary): void {
  if (!summaryLimitStatusEl) return;

  if (!limits.is_free_plan) {
    // PRO/ENTERPRISE have no per-session limits
    summaryLimitStatusEl.style.display = "none";
    return;
  }

  // FREE plan: show a subtle hint about limits
  const audioLimit = limits.per_session_audio_limit_minutes;
  if (audioLimit !== null) {
    summaryLimitStatusEl.textContent = `${t('options.summary.maxPerSession')} ${audioLimit.toFixed(0)} min/session`;
    summaryLimitStatusEl.className = "badge near"; // Use "near" style as subtle hint
    summaryLimitStatusEl.style.display = "inline-block";
    summaryLimitStatusEl.title = `FREE plan: ${audioLimit.toFixed(0)} min per session limit`;
  } else {
    summaryLimitStatusEl.style.display = "none";
  }
}

/**
 * Update the Usage & Billing section with dashboard data.
 */
function updateUsageSection(bootstrap: UserBootstrapResponse): void {
  const { usage, billing } = bootstrap.dashboard;

  // Total sessions
  setText("total-sessions", String(usage.total_sessions));

  // Audio minutes (comes as string from backend Decimal)
  const audioMinutes = parseFloat(usage.audio_minutes_total);
  setText(
    "total-audio-minutes",
    isNaN(audioMinutes) ? usage.audio_minutes_total : audioMinutes.toFixed(2)
  );

  // Transcript chars
  setText("total-transcript-chars", usage.transcript_chars_total.toLocaleString());

  // Translated chars
  setText("total-translated-chars", usage.translated_chars_total.toLocaleString());

  // Total amount
  setText("total-amount", `${billing.total_amount_eur} ${billing.currency}`);
}

// =============================================================================
// SUMMARY PANEL HELPERS
// =============================================================================

/**
 * Show loading state in the summary panel.
 */
function showSummaryLoading(): void {
  if (summaryPlanEl) summaryPlanEl.textContent = "…";
  if (summaryUsageEl) summaryUsageEl.textContent = t('options.summary.loading');
  if (summaryUpgradeBtn) summaryUpgradeBtn.style.display = "none";
  if (summaryErrorEl) summaryErrorEl.style.display = "none";
  if (summaryNoAuthEl) summaryNoAuthEl.style.display = "none";
  if (summaryContentEl) summaryContentEl.style.display = "block";
}

/**
 * Show no-auth state in the summary panel.
 */
function showSummaryNoAuth(): void {
  if (summaryContentEl) summaryContentEl.style.display = "none";
  if (summaryErrorEl) summaryErrorEl.style.display = "none";
  if (summaryNoAuthEl) summaryNoAuthEl.style.display = "block";
}

/**
 * Show error state in the summary panel.
 */
function showSummaryError(message: string): void {
  if (summaryContentEl) summaryContentEl.style.display = "none";
  if (summaryNoAuthEl) summaryNoAuthEl.style.display = "none";
  if (summaryErrorEl) {
    summaryErrorEl.textContent = message;
    summaryErrorEl.style.display = "block";
  }
}

/**
 * Render the summary panel with plan and usage data.
 * Follows the same pattern as HUD for consistency.
 */
function renderSummaryPanel(
  bootstrap: UserBootstrapResponse
): void {
  if (!summaryPlanEl || !summaryUsageEl) return;

  // Show content, hide error/no-auth
  if (summaryContentEl) summaryContentEl.style.display = "block";
  if (summaryErrorEl) summaryErrorEl.style.display = "none";
  if (summaryNoAuthEl) summaryNoAuthEl.style.display = "none";

  // Plan badge (same styling as existing plan-tier badge)
  const planTier = bootstrap.plan.plan_tier;
  summaryPlanEl.textContent = planTier;
  summaryPlanEl.className = "badge";
  if (planTier === "PRO") {
    summaryPlanEl.classList.add("pro");
  } else if (planTier === "ENTERPRISE") {
    summaryPlanEl.classList.add("enterprise");
  }

  // Usage: rounded minutes + sessions (same logic as HUD)
  const usage = bootstrap.dashboard.usage;
  const audioMinutes = parseFloat(usage.audio_minutes_total);
  const displayMinutes = isNaN(audioMinutes) ? 0 : Math.round(audioMinutes);
  const sessions = usage.total_sessions;

  let usageText = `${displayMinutes} ${t('options.summary.minutes')}`;
  if (sessions > 0) {
    usageText += ` · ${sessions} ${sessions === 1 ? t('options.summary.sessionSingular') : t('options.summary.sessions')}`;
  }
  summaryUsageEl.textContent = usageText;

  // Upgrade CTA: only show for FREE users
  if (summaryUpgradeBtn) {
    if (planTier === "FREE") {
      summaryUpgradeBtn.style.display = "inline-block";
    } else {
      summaryUpgradeBtn.style.display = "none";
    }
  }
}

// =============================================================================
// MAIN REFRESH LOGIC
// =============================================================================

/**
 * Refresh all data from the backend.
 * Calls /user/bootstrap and /user/limits-summary in parallel.
 */
async function refresh(): Promise<void> {
  const statusEl = byId<HTMLDivElement>("status");
  const errorEl = byId<HTMLDivElement>("error");
  const refreshBtn = byId<HTMLButtonElement>("refresh-btn");

  // Show loading state
  statusEl.textContent = "Loading account info…";
  setVisible("status", true);
  setVisible("error", false);
  errorEl.textContent = "";
  refreshBtn.disabled = true;

  // Show summary loading state
  showSummaryLoading();

  try {
    // Load settings
    const settings = await loadSettings();
    currentSettings = settings;

    // Apply language preference to i18n system
    // If 'auto' or missing, don't call setLocale (use auto-detection)
    if (settings.language === 'el' || settings.language === 'en') {
      setLocale(settings.language);
    }

    // Γέμισε τα Backend & Auth πεδία
    if (backendUrlInput) {
      backendUrlInput.value = settings.backendBaseUrl ?? "";
    }
    if (devTokenInput) {
      devTokenInput.value = settings.devToken ?? "";
    }
    if (debugUserIdInput) {
      debugUserIdInput.value = settings.debugUserId ?? "";
    }
    if (debugPlanInput) {
      debugPlanInput.value = settings.debugPlan ?? "";
    }

    // Render dev token expiry display
    renderDevTokenExpiry(settings);

    // Populate language selector
    if (languageSelect) {
      languageSelect.value = settings.language ?? "auto";
    }

    // Apply i18n strings to static elements now that locale is set
    applyI18nToStaticElements();

    // Populate preferences inputs
    if (sourceLangInput) {
      sourceLangInput.value = settings.defaultSourceLang ?? "";
    }
    if (targetLangInput) {
      targetLangInput.value = settings.defaultTargetLang ?? "";
    }
    if (qualitySelect) {
      qualitySelect.value = settings.defaultQualityMode ?? "FAST";
    }

    // Check if auth is configured
    if (!settings.devToken && !settings.debugUserId) {
      showSummaryNoAuth();
      // Μην κάνεις κλήσεις backend χωρίς auth — κράτα το μήνυμα no-auth στο summary
      return;
    }

    // Fetch both endpoints in parallel
    const [bootstrap, limits] = await Promise.all([
      fetchUserBootstrap(settings, 5),
      fetchUserLimitsSummary(settings),
    ]);

    // Update UI sections
    updateAccountSection(bootstrap, limits);
    updateLimitsSection(limits);
    updateUsageSection(bootstrap);

    // Update summary panel
    renderSummaryPanel(bootstrap);
    updateSummaryLimitStatus(limits);

    // Success state
    statusEl.textContent = "Loaded successfully.";
  } catch (err: unknown) {
    console.error("[options] Failed to load bootstrap/limits", err);

    // Hide status, show error
    setVisible("status", false);
    setVisible("error", true);

    // Show summary panel error
    showSummaryError(t('options.summary.loadError'));

    if (err instanceof Error) {
      errorEl.textContent = err.message;
    } else {
      errorEl.textContent =
        "Failed to load account info. Check dev token / backend.";
    }
  } finally {
    refreshBtn.disabled = false;
  }
}

// =============================================================================
// I18N HELPERS
// =============================================================================

/**
 * Apply i18n strings to static HTML elements.
 * Called after settings are loaded and locale is set.
 */
function applyI18nToStaticElements(): void {
  // Diagnostics section
  const diagnosticsTitleEl = document.getElementById("diagnostics-title");
  if (diagnosticsTitleEl) diagnosticsTitleEl.textContent = t('options.diagnostics.title');

  const diagnosticsDescEl = document.getElementById("diagnostics-description");
  if (diagnosticsDescEl) diagnosticsDescEl.textContent = t('options.diagnostics.description');

  // Buttons
  if (runDiagnosticsBtn) runDiagnosticsBtn.textContent = t('options.diagnostics.runButton');
  if (copyDiagnosticsBtn) copyDiagnosticsBtn.textContent = t('options.buttons.copyDiagnostics');
  if (testConnectionBtn) testConnectionBtn.textContent = t('options.buttons.testConnection');

  // Language selector label
  const languageLabelEl = document.getElementById("language-label");
  if (languageLabelEl && languageSelect) {
    // Update the label text (before the select element)
    const labelText = languageLabelEl.firstChild;
    if (labelText && labelText.nodeType === Node.TEXT_NODE) {
      labelText.textContent = t('options.language.label') + '\n        ';
    }
  }

  // Language selector options
  if (languageSelect) {
    const options = languageSelect.options;
    for (let i = 0; i < options.length; i++) {
      const opt = options[i];
      if (opt.value === 'auto') opt.textContent = t('options.language.auto');
      else if (opt.value === 'el') opt.textContent = t('options.language.el');
      else if (opt.value === 'en') opt.textContent = t('options.language.en');
    }
  }

  // Summary no-auth message
  const summaryNoAuthEl = document.getElementById("summary-no-auth");
  if (summaryNoAuthEl) summaryNoAuthEl.textContent = t('options.summary.noAuth');

  // Shortcuts & controls section
  const shortcutsSectionTitleEl = document.getElementById("shortcuts-section-title");
  if (shortcutsSectionTitleEl) shortcutsSectionTitleEl.textContent = t('options.shortcuts.sectionTitle');

  const shortcutsHowToStartEl = document.getElementById("shortcuts-how-to-start");
  if (shortcutsHowToStartEl) shortcutsHowToStartEl.textContent = t('options.shortcuts.howToStart');

  const shortcutsHowToStartStepsEl = document.getElementById("shortcuts-how-to-start-steps");
  if (shortcutsHowToStartStepsEl) shortcutsHowToStartStepsEl.textContent = t('options.shortcuts.howToStart.steps');

  const shortcutsKeyboardHeadingEl = document.getElementById("shortcuts-keyboard-heading");
  if (shortcutsKeyboardHeadingEl) shortcutsKeyboardHeadingEl.textContent = t('options.shortcuts.keyboardHeading');

  const shortcutsKeyboardToggleEl = document.getElementById("shortcuts-keyboard-toggle");
  if (shortcutsKeyboardToggleEl) shortcutsKeyboardToggleEl.textContent = t('options.shortcuts.keyboardToggle');

  const shortcutsKeyboardOpenOptionsEl = document.getElementById("shortcuts-keyboard-open-options");
  if (shortcutsKeyboardOpenOptionsEl) shortcutsKeyboardOpenOptionsEl.textContent = t('options.shortcuts.keyboardOpenOptions');

  const shortcutsContextMenuHeadingEl = document.getElementById("shortcuts-context-menu-heading");
  if (shortcutsContextMenuHeadingEl) shortcutsContextMenuHeadingEl.textContent = t('options.shortcuts.contextMenuHeading');

  const shortcutsContextMenuDescriptionEl = document.getElementById("shortcuts-context-menu-description");
  if (shortcutsContextMenuDescriptionEl) shortcutsContextMenuDescriptionEl.textContent = t('options.shortcuts.contextMenuDescription');

  const shortcutsChromeCustomizeEl = document.getElementById("shortcuts-chrome-customize");
  if (shortcutsChromeCustomizeEl) shortcutsChromeCustomizeEl.textContent = t('options.shortcuts.chromeShortcutsNote');

  // Help & troubleshooting section
  const helpSectionTitleEl = document.getElementById("help-section-title");
  if (helpSectionTitleEl) helpSectionTitleEl.textContent = t('options.help.sectionTitle');

  const helpWhenToUseEl = document.getElementById("help-when-to-use");
  if (helpWhenToUseEl) helpWhenToUseEl.textContent = t('options.help.whenToUseDiagnostics');

  const helpStepsHeadingEl = document.getElementById("help-steps-heading");
  if (helpStepsHeadingEl) helpStepsHeadingEl.textContent = t('options.help.stepsHeading');

  const helpStepCheckBackendEl = document.getElementById("help-step-check-backend");
  if (helpStepCheckBackendEl) helpStepCheckBackendEl.textContent = t('options.help.stepCheckBackend');

  const helpStepCheckMicEl = document.getElementById("help-step-check-mic");
  if (helpStepCheckMicEl) helpStepCheckMicEl.textContent = t('options.help.stepCheckMic');

  const helpStepCopyReportEl = document.getElementById("help-step-copy-report");
  if (helpStepCopyReportEl) helpStepCopyReportEl.textContent = t('options.help.stepCopyReport');

  const helpPrivacyNoteEl = document.getElementById("help-privacy-note");
  if (helpPrivacyNoteEl) helpPrivacyNoteEl.textContent = t('options.help.privacyNote');
}

// =============================================================================
// INITIALIZATION
// =============================================================================

/**
 * Initialize the options page.
 * Sets up event listeners and triggers initial data load.
 */
function initOptionsPage(): void {
  const refreshBtn = byId<HTMLButtonElement>("refresh-btn");

  // Get preferences elements
  sourceLangInput = document.getElementById("llp-source-lang") as HTMLInputElement | null;
  targetLangInput = document.getElementById("llp-target-lang") as HTMLInputElement | null;
  qualitySelect = document.getElementById("llp-quality-mode") as HTMLSelectElement | null;
  languageSelect = document.getElementById("language-select") as HTMLSelectElement | null;
  saveBtn = document.getElementById("save-btn") as HTMLButtonElement | null;
  saveStatusEl = document.getElementById("save-status") as HTMLSpanElement | null;

  // Get backend & auth elements
  backendUrlInput = document.getElementById("llp-backend-url") as HTMLInputElement | null;
  devTokenInput = document.getElementById("llp-dev-token") as HTMLInputElement | null;
  debugUserIdInput = document.getElementById("llp-debug-user-id") as HTMLInputElement | null;
  debugPlanInput = document.getElementById("llp-debug-plan") as HTMLInputElement | null;
  testConnectionBtn = document.getElementById("test-connection-btn") as HTMLButtonElement | null;
  connectionTestStatusEl = document.getElementById("connection-test-status") as HTMLSpanElement | null;
  generateDevTokenBtn = document.getElementById("generate-dev-token-btn") as HTMLButtonElement | null;
  generateDevTokenStatusEl = document.getElementById("generate-dev-token-status") as HTMLDivElement | null;
  revokeDevTokenBtn = document.getElementById("revoke-dev-token-btn") as HTMLButtonElement | null;
  devTokenExpiryEl = document.getElementById("dev-token-expiry") as HTMLParagraphElement | null;

  // Get summary panel elements
  summaryPlanEl = document.getElementById("summary-plan") as HTMLSpanElement | null;
  summaryUsageEl = document.getElementById("summary-usage") as HTMLSpanElement | null;
  summaryUpgradeBtn = document.getElementById("summary-upgrade-btn") as HTMLAnchorElement | null;
  summaryErrorEl = document.getElementById("summary-error") as HTMLDivElement | null;
  summaryNoAuthEl = document.getElementById("summary-no-auth") as HTMLDivElement | null;
  summaryContentEl = document.getElementById("summary-content") as HTMLDivElement | null;
  summaryLimitStatusEl = document.getElementById("summary-limit-status") as HTMLSpanElement | null;

  // Get limit status elements
  audioLimitStatusEl = document.getElementById("audio-limit-status") as HTMLSpanElement | null;
  translationLimitStatusEl = document.getElementById("translation-limit-status") as HTMLSpanElement | null;

  // Get diagnostics elements
  runDiagnosticsBtn = document.getElementById("run-diagnostics-btn") as HTMLButtonElement | null;
  diagnosticsResultsEl = document.getElementById("diagnostics-results") as HTMLDivElement | null;
  diagnosticsListEl = document.getElementById("diagnostics-list") as HTMLDivElement | null;
  copyDiagnosticsBtn = document.getElementById("copy-diagnostics-btn") as HTMLButtonElement | null;
  copyStatusEl = document.getElementById("copy-status") as HTMLSpanElement | null;

  // Wire up refresh button
  refreshBtn.addEventListener("click", () => {
    void refresh();
  });

  // Wire up save button
  if (saveBtn) {
    saveBtn.addEventListener("click", () => {
      void handleSave();
    });
  }

  // Wire up test connection button
  if (testConnectionBtn) {
    testConnectionBtn.addEventListener("click", () => {
      void handleTestConnection();
    });
  }

  // Wire up generate dev token button
  if (generateDevTokenBtn) {
    generateDevTokenBtn.textContent = t('options.backend.generateDevToken.button');
    generateDevTokenBtn.addEventListener("click", () => {
      void handleGenerateDevToken();
    });
  }

  // Wire up revoke dev token button
  if (revokeDevTokenBtn) {
    revokeDevTokenBtn.textContent = t('options.backend.revokeDevToken.button');
    revokeDevTokenBtn.addEventListener("click", () => {
      void handleRevokeDevToken();
    });
  }

  // Wire up diagnostics buttons
  if (runDiagnosticsBtn) {
    runDiagnosticsBtn.addEventListener("click", () => {
      void handleRunDiagnostics();
    });
  }
  if (copyDiagnosticsBtn) {
    copyDiagnosticsBtn.addEventListener("click", () => {
      void handleCopyDiagnostics();
    });
  }

  // Initial load
  void refresh();
}

/**
 * Handle testing connection to the backend.
 * Saves current form values first, then tests with fetchUserBootstrap.
 */
async function handleTestConnection(): Promise<void> {
  if (!testConnectionBtn || !connectionTestStatusEl) return;

  testConnectionBtn.disabled = true;
  connectionTestStatusEl.textContent = t('options.connection.testing');
  connectionTestStatusEl.style.color = "#666";

  try {
    // First, save the current form values to settings
    const updates: Partial<LinguaLiveSettings> = {};

    if (backendUrlInput) {
      const v = backendUrlInput.value.trim();
      updates.backendBaseUrl = v || undefined;
    }
    if (devTokenInput) {
      const v = devTokenInput.value.trim();
      updates.devToken = v || null;
    }
    if (debugUserIdInput) {
      const v = debugUserIdInput.value.trim();
      updates.debugUserId = v || null;
    }
    if (debugPlanInput) {
      const v = debugPlanInput.value.trim().toUpperCase();
      if (v === "FREE" || v === "PRO" || v === "ENTERPRISE") {
        updates.debugPlan = v;
      } else {
        updates.debugPlan = null;
      }
    }

    await saveSettings(updates);
    const settings = await loadSettings();

    // Check if auth is configured
    if (!settings.devToken && !settings.debugUserId) {
      connectionTestStatusEl.textContent = t('options.connection.noAuth');
      connectionTestStatusEl.style.color = "#b00020";
      return;
    }

    // Test connection by calling bootstrap endpoint
    const bootstrap = await fetchUserBootstrap(settings, 5);

    // Success!
    const planTier = bootstrap.plan.plan_tier;
    connectionTestStatusEl.textContent = `${t('options.connection.success')} ${bootstrap.whoami.user_id} (${planTier})`;
    connectionTestStatusEl.style.color = "#2e7d32";

  } catch (err) {
    console.error("[options] Connection test failed", err);

    // Detect auth errors
    const statusCode = (err as { status?: number }).status;
    const message = err instanceof Error ? err.message : String(err);

    if (statusCode === 401 || statusCode === 403) {
      connectionTestStatusEl.textContent = t('options.connection.invalidToken');
    } else if (message.includes("fetch") || message.includes("network") || message.includes("Failed to fetch")) {
      connectionTestStatusEl.textContent = t('options.connection.serverNotFound');
    } else {
      connectionTestStatusEl.textContent = `❌ ${message}`;
    }
    connectionTestStatusEl.style.color = "#b00020";
  } finally {
    testConnectionBtn.disabled = false;
  }
}

/**
 * Render the dev token expiry display.
 * Shows expiry date/time or "no active token" message.
 */
function renderDevTokenExpiry(settings: LinguaLiveSettings): void {
  if (!devTokenExpiryEl) return;

  if (!settings.devToken || !settings.devTokenExpiresAt) {
    devTokenExpiryEl.textContent = t('options.backend.devTokenExpiry.none');
    devTokenExpiryEl.style.color = "#666";
    return;
  }

  // Format the expiry date for display
  try {
    const expiryDate = new Date(settings.devTokenExpiresAt);
    const formatted = expiryDate.toLocaleString(undefined, {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      timeZoneName: 'short',
    });
    devTokenExpiryEl.textContent = `${t('options.backend.devTokenExpiry.label')} ${formatted}`;
    devTokenExpiryEl.style.color = "#333";
  } catch {
    // Fallback to raw ISO string if parsing fails
    devTokenExpiryEl.textContent = `${t('options.backend.devTokenExpiry.label')} ${settings.devTokenExpiresAt}`;
    devTokenExpiryEl.style.color = "#333";
  }
}

/**
 * Handle generating a new dev token.
 * Calls POST /dev/tokens using debug headers, then saves the returned token.
 */
async function handleGenerateDevToken(): Promise<void> {
  if (!generateDevTokenBtn || !generateDevTokenStatusEl) return;

  generateDevTokenBtn.disabled = true;
  generateDevTokenStatusEl.textContent = t('options.backend.generateDevToken.generating');
  generateDevTokenStatusEl.style.color = "#666";

  try {
    // First, save the current form values to settings
    const updates: Partial<LinguaLiveSettings> = {};

    if (backendUrlInput) {
      const v = backendUrlInput.value.trim();
      updates.backendBaseUrl = v || undefined;
    }
    if (debugUserIdInput) {
      const v = debugUserIdInput.value.trim();
      updates.debugUserId = v || null;
    }
    if (debugPlanInput) {
      const v = debugPlanInput.value.trim().toUpperCase();
      if (v === "FREE" || v === "PRO" || v === "ENTERPRISE") {
        updates.debugPlan = v;
      } else {
        updates.debugPlan = null;
      }
    }

    await saveSettings(updates);
    const settings = await loadSettings();

    // Validate backend URL
    if (!settings.backendBaseUrl) {
      generateDevTokenStatusEl.textContent = t('options.backend.generateDevToken.errorMissingBackend');
      generateDevTokenStatusEl.style.color = "#b00020";
      return;
    }

    // Validate debug user ID
    if (!settings.debugUserId) {
      generateDevTokenStatusEl.textContent = t('options.backend.generateDevToken.errorMissingUser');
      generateDevTokenStatusEl.style.color = "#b00020";
      return;
    }

    // Request new dev token
    const tokenResponse = await requestDevToken(settings.backendBaseUrl, settings, 60);

    // Save the token and expiry to settings
    settings.devToken = tokenResponse.token;
    settings.devTokenExpiresAt = tokenResponse.expires_at ?? null;
    await saveSettings(settings);

    // Update the dev token input field
    if (devTokenInput) {
      devTokenInput.value = tokenResponse.token;
    }

    // Update the expiry display
    renderDevTokenExpiry(settings);

    // Success!
    generateDevTokenStatusEl.textContent = t('options.backend.generateDevToken.success');
    generateDevTokenStatusEl.style.color = "#2e7d32";

  } catch (err) {
    console.error("[options] Generate dev token failed", err);

    const message = err instanceof Error ? err.message : String(err);

    if (message === "missing_debug_user_id") {
      generateDevTokenStatusEl.textContent = t('options.backend.generateDevToken.errorMissingUser');
    } else if (message.startsWith("dev_token_http_")) {
      generateDevTokenStatusEl.textContent = t('options.backend.generateDevToken.errorHttp');
    } else if (message.includes("fetch") || message.includes("network") || message.includes("Failed to fetch")) {
      generateDevTokenStatusEl.textContent = t('options.backend.generateDevToken.errorNetwork');
    } else {
      generateDevTokenStatusEl.textContent = t('options.backend.generateDevToken.errorNetwork');
    }
    generateDevTokenStatusEl.style.color = "#b00020";
  } finally {
    generateDevTokenBtn.disabled = false;
  }
}

/**
 * Handle revoking the current dev token.
 * Calls POST /dev/tokens/revoke, then clears devToken + devTokenExpiresAt from settings.
 */
async function handleRevokeDevToken(): Promise<void> {
  if (!revokeDevTokenBtn || !generateDevTokenStatusEl) return;

  revokeDevTokenBtn.disabled = true;
  generateDevTokenStatusEl.textContent = t('options.backend.revokeDevToken.revoking');
  generateDevTokenStatusEl.style.color = "#666";

  try {
    const settings = await loadSettings();

    // Validate backend URL
    if (!settings.backendBaseUrl) {
      generateDevTokenStatusEl.textContent = t('options.backend.generateDevToken.errorMissingBackend');
      generateDevTokenStatusEl.style.color = "#b00020";
      return;
    }

    // Validate dev token exists
    if (!settings.devToken) {
      generateDevTokenStatusEl.textContent = t('options.backend.revokeDevToken.errorNoToken');
      generateDevTokenStatusEl.style.color = "#b00020";
      return;
    }

    // Call revoke endpoint
    await revokeDevToken(settings.backendBaseUrl, settings);

    // Clear token and expiry from settings (regardless of revoked=true/false)
    settings.devToken = null;
    settings.devTokenExpiresAt = null;
    await saveSettings(settings);

    // Update UI
    if (devTokenInput) {
      devTokenInput.value = "";
    }
    renderDevTokenExpiry(settings);

    // Success!
    generateDevTokenStatusEl.textContent = t('options.backend.revokeDevToken.success');
    generateDevTokenStatusEl.style.color = "#2e7d32";

  } catch (err) {
    console.error("[options] Revoke dev token failed", err);

    const message = err instanceof Error ? err.message : String(err);

    if (message === "missing_dev_token") {
      generateDevTokenStatusEl.textContent = t('options.backend.revokeDevToken.errorNoToken');
    } else if (message.startsWith("dev_token_revoke_http_")) {
      generateDevTokenStatusEl.textContent = t('options.backend.revokeDevToken.errorHttp');
    } else if (message.includes("fetch") || message.includes("network") || message.includes("Failed to fetch")) {
      generateDevTokenStatusEl.textContent = t('options.backend.revokeDevToken.errorNetwork');
    } else {
      generateDevTokenStatusEl.textContent = t('options.backend.revokeDevToken.errorNetwork');
    }
    generateDevTokenStatusEl.style.color = "#b00020";
  } finally {
    revokeDevTokenBtn.disabled = false;
  }
}

/**
 * Handle saving preferences.
 */
async function handleSave(): Promise<void> {
  if (!saveBtn || !saveStatusEl) return;

  saveBtn.disabled = true;
  saveStatusEl.textContent = "Saving…";

  try {
    const updates: Partial<LinguaLiveSettings> = {};

    // Backend URL & Auth headers
    // Αυτά τα πεδία καθορίζουν πού θα συνδεθεί το extension και με ποια auth.
    if (backendUrlInput) {
      const v = backendUrlInput.value.trim();
      updates.backendBaseUrl = v || undefined;
    }
    if (devTokenInput) {
      const v = devTokenInput.value.trim();
      updates.devToken = v || null;
    }
    if (debugUserIdInput) {
      const v = debugUserIdInput.value.trim();
      updates.debugUserId = v || null;
    }
    if (debugPlanInput) {
      const v = debugPlanInput.value.trim().toUpperCase();
      // Ελέγχουμε αν είναι έγκυρο plan tier
      if (v === "FREE" || v === "PRO" || v === "ENTERPRISE") {
        updates.debugPlan = v;
      } else {
        updates.debugPlan = null;
      }
    }

    // Preferences (languages/quality)
    // Αυτά είναι οι προεπιλογές για realtime sessions.
    if (sourceLangInput) {
      const v = sourceLangInput.value.trim();
      updates.defaultSourceLang = v || undefined;
    }
    if (targetLangInput) {
      const v = targetLangInput.value.trim();
      updates.defaultTargetLang = v || undefined;
    }
    if (qualitySelect) {
      const v = qualitySelect.value as "FAST" | "SMART";
      updates.defaultQualityMode = v;
    }

    // UI Language preference
    if (languageSelect) {
      const v = languageSelect.value as "auto" | "el" | "en";
      updates.language = v;
      // Apply immediately to i18n system
      if (v === 'el' || v === 'en') {
        setLocale(v);
      }
    }

    await saveSettings(updates);

    saveStatusEl.textContent = "Saved!";
    setTimeout(() => {
      if (saveStatusEl) saveStatusEl.textContent = "";
    }, 2000);
  } catch (err) {
    console.error("[options] Failed to save preferences", err);
    saveStatusEl.textContent = "Failed to save.";
  } finally {
    saveBtn.disabled = false;
  }
}

// Wait for DOM ready
document.addEventListener("DOMContentLoaded", () => {
  initOptionsPage();
});
