/**
 * =============================================================================
 * BACKENDCLIENT.TS — Typed HTTP Client for LinguaLive Pro Backend
 * =============================================================================
 *
 * Παρέχει typed API client για τα user endpoints του backend:
 * - GET /user/bootstrap — Unified startup data (whoami + plan + dashboard)
 * - GET /user/limits-summary — Plan limits και features
 *
 * ΧΡΗΣΗ:
 * ------
 *   import { fetchUserBootstrap, fetchUserLimitsSummary } from "./api/backendClient";
 *   import { loadSettings } from "./storage/settings";
 *
 *   const settings = await loadSettings();
 *   const bootstrap = await fetchUserBootstrap(settings);
 *   console.log(bootstrap.whoami.user_id);
 *
 * AUTH:
 * -----
 * Χρησιμοποιεί την ίδια λογική με τον backend:
 * 1. Αν υπάρχει devToken → x-dev-token header (προτεραιότητα)
 * 2. Αλλιώς → x-debug-user-id + x-debug-plan headers
 *
 * ΣΗΜΕΙΩΣΕΙΣ:
 * -----------
 * - Αυτό είναι dev-only client. Σε production θα αλλάξει το auth layer.
 * - Τα types αντιστοιχούν στα Pydantic models του backend (UserBootstrapResponse, UserLimitsSummary).
 * =============================================================================
 */

import {
  type LinguaLiveSettings,
  type PlanTier,
  type QualityMode,
  getApiBaseUrl,
} from "../storage/settings";

// =============================================================================
// TYPES — Response Models (match backend Pydantic models)
// =============================================================================

/**
 * User identity snapshot.
 * Matches backend UserWhoAmI model.
 */
export interface UserWhoAmI {
  user_id: string;
  plan_tier: PlanTier;
  auth_source: "dev_token" | "debug_headers";
  dev_token_expires_at: string | null;
}

/**
 * Pricing info for a plan.
 * Subset of backend PublicPlanPricing.
 */
export interface PublicPlanPricing {
  audio_price_per_minute_eur: string;
  translation_price_per_kchar_eur: string;
}

/**
 * Plan limits.
 * Subset of backend PublicPlanLimits.
 */
export interface PublicPlanLimits {
  max_session_audio_minutes: number | null;
  max_session_translated_chars: number | null;
}

/**
 * Plan features.
 * Subset of backend PublicPlanFeatures.
 */
export interface PublicPlanFeatures {
  allowed_quality_modes: QualityMode[];
}

/**
 * Full plan info.
 * Matches backend PublicPlanInfo model.
 */
export interface PublicPlanInfo {
  plan_tier: PlanTier;
  display_name: string;
  pricing: PublicPlanPricing;
  limits: PublicPlanLimits;
  features: PublicPlanFeatures;
}

/**
 * Usage summary.
 * Matches backend PublicUserUsageSummary.
 */
export interface PublicUserUsageSummary {
  user_id: string;
  total_sessions: number;
  audio_ms_total: number;
  /**
   * Serialized decimal string from backend (Decimal) — do not assume number math without parsing.
   */
  audio_minutes_total: string;
  transcript_chars_total: number;
  translated_chars_total: number;
}

/**
 * Billing summary.
 * Matches backend PublicUserBillingSummary.
 */
export interface PublicUserBillingSummary {
  user_id: string;
  total_sessions_billed: number;
  total_amount_eur: string;
  audio_amount_eur: string;
  translation_amount_eur: string;
  currency: string;
}

/**
 * Session entry for recent sessions list.
 * Matches backend UserSessionEntry.
 */
export interface UserSessionEntry {
  session_id: string;
  started_at: string;
  ended_at: string | null;
  duration_seconds: number | null;
  plan_tier: PlanTier;
  quality_mode: QualityMode;
  source_lang: string;
  target_lang: string;
  audio_ms_total: number;
  transcript_chars_total: number;
  translated_chars_total: number;
  total_amount_eur: string | null;
  currency: string | null;
}

/**
 * Dashboard summary combining usage, billing, and recent sessions.
 * Matches backend UserDashboardSummary.
 */
export interface UserDashboardSummary {
  user_id: string;
  usage: PublicUserUsageSummary;
  billing: PublicUserBillingSummary;
  recent_sessions: UserSessionEntry[];
}

/**
 * Bootstrap response — unified startup data.
 * Matches backend UserBootstrapResponse.
 */
export interface UserBootstrapResponse {
  whoami: UserWhoAmI;
  plan: PublicPlanInfo;
  dashboard: UserDashboardSummary;
}

/**
 * Limits summary — user-centered view of plan limits.
 * Matches backend UserLimitsSummary.
 */
export interface UserLimitsSummary {
  user_id: string;
  plan_tier: PlanTier;
  is_free_plan: boolean;
  allowed_quality_modes: QualityMode[];
  per_session_audio_limit_minutes: number | null;
  per_session_translation_chars_limit: number | null;
}

/**
 * Public status response from GET /public/status.
 * Matches backend PublicStatusInfo model.
 */
export interface PublicStatusResponse {
  service: string;
  version: string;
  status: string;
  environment: string;
  pipeline_latency_ms: number;
  now_utc: string;
}

/**
 * Dev token response from POST /dev/tokens.
 * Matches backend DevSessionToken model.
 */
export interface DevTokenResponse {
  token: string;
  user_id: string;
  plan_tier: PlanTier;
  created_at: string;
  expires_at: string;
}

/**
 * Dev token revoke response from POST /dev/tokens/revoke.
 * Matches backend DevTokenRevokeResponse model.
 */
export interface DevTokenRevokeResponse {
  revoked: boolean;
}

// =============================================================================
// AUTH HEADER BUILDER
// =============================================================================

/**
 * Type for auth headers.
 */
type AuthHeaders = Record<string, string>;

/**
 * Build authentication headers based on settings.
 *
 * Priority:
 * 1. If devToken exists → x-dev-token header (takes priority)
 * 2. Otherwise → x-debug-user-id + optional x-debug-plan
 *
 * This matches the backend's auth resolution logic.
 *
 * @param settings - Extension settings containing auth config
 * @returns Headers object with appropriate auth headers
 * @throws Error if no auth credentials are available
 */
export function buildAuthHeaders(settings: LinguaLiveSettings): AuthHeaders {
  const headers: AuthHeaders = {};

  // Priority 1: Dev token (if present)
  if (settings.devToken) {
    headers["x-dev-token"] = settings.devToken;
    return headers;
  }

  // Priority 2: Debug headers (fallback)
  if (settings.debugUserId) {
    headers["x-debug-user-id"] = settings.debugUserId;

    // Add plan header if specified
    if (settings.debugPlan) {
      headers["x-debug-plan"] = settings.debugPlan;
    }
    return headers;
  }

  // No auth credentials available
  throw new Error(
    "No auth credentials configured. Set either devToken or debugUserId in settings."
  );
}

// =============================================================================
// ERROR HANDLING
// =============================================================================

/**
 * API error with status code and detail.
 */
export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail: string,
    public readonly endpoint: string
  ) {
    super(`${endpoint} failed: ${status} ${detail}`);
    this.name = "ApiError";
  }
}

/**
 * Parse error response from backend.
 * Backend returns { "detail": "..." } for errors.
 */
async function parseErrorResponse(response: Response): Promise<string> {
  try {
    const json = await response.json();
    if (typeof json.detail === "string") {
      return json.detail;
    }
    // Sometimes detail is an object with more info
    if (typeof json.detail === "object" && json.detail !== null) {
      return JSON.stringify(json.detail);
    }
    return response.statusText;
  } catch {
    return response.statusText;
  }
}

// =============================================================================
// FETCH HELPERS
// =============================================================================

/**
 * Fetch user bootstrap data.
 *
 * Retrieves unified startup data including:
 * - whoami (identity snapshot)
 * - plan (full plan info with pricing, limits, features)
 * - dashboard (usage, billing, recent sessions)
 *
 * @param settings - Extension settings with backend URL and auth
 * @param sessionLimit - Max recent sessions to include (default 5, max 20)
 * @returns UserBootstrapResponse with all startup data
 * @throws ApiError if request fails
 *
 * @example
 * const bootstrap = await fetchUserBootstrap(settings);
 * console.log(`User: ${bootstrap.whoami.user_id}`);
 * console.log(`Plan: ${bootstrap.plan.display_name}`);
 * console.log(`Sessions: ${bootstrap.dashboard.recent_sessions.length}`);
 */
export async function fetchUserBootstrap(
  settings: LinguaLiveSettings,
  sessionLimit: number = 5
): Promise<UserBootstrapResponse> {
  const baseUrl = getApiBaseUrl(settings);
  const url = `${baseUrl}/user/bootstrap?session_limit=${sessionLimit}`;

  const headers: HeadersInit = {
    Accept: "application/json",
    "Content-Type": "application/json",
    ...buildAuthHeaders(settings),
  };

  const response = await fetch(url, {
    method: "GET",
    headers,
  });

  if (!response.ok) {
    const detail = await parseErrorResponse(response);
    throw new ApiError(response.status, detail, "/user/bootstrap");
  }

  return response.json() as Promise<UserBootstrapResponse>;
}

/**
 * Fetch user limits summary.
 *
 * Retrieves plan limits and features for the current user:
 * - allowed_quality_modes (which modes the user can use)
 * - per_session_audio_limit_minutes (null = unlimited)
 * - per_session_translation_chars_limit (null = unlimited)
 * - is_free_plan (for upgrade prompts)
 *
 * @param settings - Extension settings with backend URL and auth
 * @returns UserLimitsSummary with limits and features
 * @throws ApiError if request fails
 *
 * @example
 * const limits = await fetchUserLimitsSummary(settings);
 * if (limits.is_free_plan) {
 *   showUpgradePrompt();
 * }
 * if (!limits.allowed_quality_modes.includes("SMART")) {
 *   disableSmartMode();
 * }
 */
export async function fetchUserLimitsSummary(
  settings: LinguaLiveSettings
): Promise<UserLimitsSummary> {
  const baseUrl = getApiBaseUrl(settings);
  const url = `${baseUrl}/user/limits-summary`;

  const headers: HeadersInit = {
    Accept: "application/json",
    "Content-Type": "application/json",
    ...buildAuthHeaders(settings),
  };

  const response = await fetch(url, {
    method: "GET",
    headers,
  });

  if (!response.ok) {
    const detail = await parseErrorResponse(response);
    throw new ApiError(response.status, detail, "/user/limits-summary");
  }

  return response.json() as Promise<UserLimitsSummary>;
}

/**
 * Fetch user usage summary.
 *
 * Retrieves aggregated usage statistics for the current user:
 * - total_sessions: Number of sessions
 * - audio_ms_total: Total audio in milliseconds
 * - audio_minutes_total: Total audio in minutes (string from Decimal)
 * - transcript_chars_total: Total transcribed characters
 * - translated_chars_total: Total translated characters
 *
 * @param settings - Extension settings with backend URL and auth
 * @returns PublicUserUsageSummary with usage stats
 * @throws ApiError if request fails
 *
 * @example
 * const usage = await fetchUserUsageSummary(settings);
 * console.log(`Total audio: ${usage.audio_minutes_total} min`);
 */
export async function fetchUserUsageSummary(
  settings: LinguaLiveSettings
): Promise<PublicUserUsageSummary> {
  const baseUrl = getApiBaseUrl(settings);
  const url = `${baseUrl}/user/usage-summary`;

  const headers: HeadersInit = {
    Accept: "application/json",
    "Content-Type": "application/json",
    ...buildAuthHeaders(settings),
  };

  const response = await fetch(url, {
    method: "GET",
    headers,
  });

  if (!response.ok) {
    const detail = await parseErrorResponse(response);
    throw new ApiError(response.status, detail, "/user/usage-summary");
  }

  return response.json() as Promise<PublicUserUsageSummary>;
}

/**
 * Fetch public backend status.
 *
 * Calls the public health endpoint GET /public/status which does not
 * require authentication. Useful for diagnostics to check if the
 * backend is reachable and what environment/version it runs.
 *
 * @param backendBaseUrl - Base URL of the backend (e.g., "http://localhost:8000")
 * @param timeoutSeconds - Optional timeout in seconds (default: 5)
 * @returns PublicStatusResponse with service info
 * @throws Error if request fails or times out
 *
 * @example
 * const status = await fetchPublicStatus("http://localhost:8000");
 * console.log(`Backend: ${status.service} v${status.version} (${status.environment})`);
 */
export async function fetchPublicStatus(
  backendBaseUrl: string,
  timeoutSeconds: number = 5
): Promise<PublicStatusResponse> {
  const url = new URL("/public/status", backendBaseUrl).toString();

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutSeconds * 1000);

  try {
    const response = await fetch(url, {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`Status endpoint returned ${response.status}`);
    }

    return (await response.json()) as PublicStatusResponse;
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Request a new dev token from POST /dev/tokens.
 *
 * Uses debug headers (x-debug-user-id, x-debug-plan) to issue a token.
 * The returned token can then be used for authenticated requests.
 *
 * @param backendBaseUrl - Base URL of the backend
 * @param settings - Extension settings with debug credentials
 * @param ttlMinutes - Token TTL in minutes (default: 60)
 * @returns DevTokenResponse with the new token
 * @throws Error if debug user ID is missing or request fails
 *
 * @example
 * const token = await requestDevToken("http://localhost:8000", settings);
 * settings.devToken = token.token;
 * await saveSettings(settings);
 */
export async function requestDevToken(
  backendBaseUrl: string,
  settings: LinguaLiveSettings,
  ttlMinutes: number = 60
): Promise<DevTokenResponse> {
  if (!settings.debugUserId) {
    throw new Error("missing_debug_user_id");
  }

  const url = new URL("/dev/tokens", backendBaseUrl);
  url.searchParams.set("ttl_minutes", String(ttlMinutes));

  const headers: HeadersInit = {
    "x-debug-user-id": settings.debugUserId,
    Accept: "application/json",
  };

  if (settings.debugPlan) {
    headers["x-debug-plan"] = settings.debugPlan;
  }

  const response = await fetch(url.toString(), {
    method: "POST",
    headers,
  });

  if (!response.ok) {
    throw new Error(`dev_token_http_${response.status}`);
  }

  return (await response.json()) as DevTokenResponse;
}

/**
 * Revoke an existing dev token via POST /dev/tokens/revoke.
 *
 * Uses the x-dev-token header to identify the token to revoke.
 * Returns { revoked: true } if token was found and revoked,
 * { revoked: false } if token didn't exist or was already revoked.
 *
 * @param backendBaseUrl - Base URL of the backend
 * @param settings - Extension settings with the devToken to revoke
 * @returns DevTokenRevokeResponse with revoked status
 * @throws Error if devToken is missing or request fails
 *
 * @example
 * const result = await revokeDevToken("http://localhost:8000", settings);
 * if (result.revoked) {
 *   settings.devToken = null;
 *   await saveSettings(settings);
 * }
 */
export async function revokeDevToken(
  backendBaseUrl: string,
  settings: LinguaLiveSettings
): Promise<DevTokenRevokeResponse> {
  if (!settings.devToken) {
    throw new Error("missing_dev_token");
  }

  const url = new URL("/dev/tokens/revoke", backendBaseUrl);

  const response = await fetch(url.toString(), {
    method: "POST",
    headers: {
      "x-dev-token": settings.devToken,
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`dev_token_revoke_http_${response.status}`);
  }

  return (await response.json()) as DevTokenRevokeResponse;
}
