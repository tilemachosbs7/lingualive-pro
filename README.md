# LinguaLive Pro Extension

Real-time speech-to-text captions and translation overlay for any browser tab.

## Overview

The LinguaLive Pro browser extension provides:

- **Real-time captions** — Transcribes speech from your microphone or tab audio
- **Translation overlay** — Translates captions to your target language in real-time
- **HUD overlay** — Floating control panel with start/stop, plan info, usage stats, and status
- **Options page** — Configure backend, authentication, language, and run diagnostics

Works with a LinguaLive Pro backend account (FREE / PRO / ENTERPRISE plans).

### Key UI Surfaces

| Surface | Description |
|---------|-------------|
| **HUD Overlay** | Floating panel on any tab — start/stop captions, view plan, usage, limits, quality mode, status |
| **Options Page** | Backend URL, API token, Account & Usage summary, Plan Limits, Diagnostics, Language settings |

---

## Installation (Development Mode)

The extension is not yet available in the Chrome Web Store. To install locally for development:

### 1. Build the Extension

```bash
cd extension
npm install
npm run build
```

This generates the compiled JavaScript in `dist/`.

### 2. Load in Chrome

1. Open Chrome and navigate to: `chrome://extensions`
2. Enable **Developer mode** (toggle in top-right corner)
3. Click **"Load unpacked"**
4. Select the `extension/` folder (the root folder containing `manifest.json`)
5. Verify that **"LinguaLive Pro (Dev)"** appears in your extensions list

### Verification

- ✅ Extension appears in the extensions list
- ✅ Service Worker shows "Active" or has an "Inspect" link
- ✅ No errors in the extension card

---

## Configuration

### Opening Options

- Click the extension icon → **"Details"** → **"Extension options"**
- Or navigate directly to: `chrome-extension://<extension-id>/options.html`

### Backend & Auth Section

| Field | Description |
|-------|-------------|
| **Backend base URL** | The URL of your LinguaLive backend (e.g., `http://localhost:8000`) |
| **Dev token (x-dev-token)** | API token from the `/dev/tokens` endpoint. Takes priority over debug headers |
| **Debug user id** | Fallback if no dev token is set (header: `x-debug-user-id`) |
| **Debug plan** | Fallback plan tier for testing (FREE / PRO / ENTERPRISE) |

### Test Connection Button

The **"Έλεγχος σύνδεσης" / "Test connection"** button:

- Saves your current settings
- Calls the backend's `/user/bootstrap` endpoint
- Shows the result:
  - **✅ Συνδέθηκες ως...** — Success, shows your user ID and plan
  - **❌ Μη έγκυρο token...** — Token is invalid or expired (401/403)
  - **❌ Δεν βρέθηκε ο server...** — Backend URL is wrong or server is down
  - **❌ Δεν έχεις ορίσει token...** — No token or debug user configured

### Account & Usage Summary Panel

Displays your account information at a glance:

| Element | Description |
|---------|-------------|
| **Plan Badge** | Shows FREE, PRO, or ENTERPRISE |
| **Usage** | Total minutes and sessions used |
| **Limit Status** | For FREE plans, shows per-session limit hint (e.g., "Max 5 min/session") |
| **Upgrade CTA** | Link to pricing page (shown for FREE users) |

### Plan Limits Section

Shows your per-session limits:

| Limit | Description |
|-------|-------------|
| **Per-session audio limit** | Maximum audio minutes per session (e.g., "5.0 minutes") |
| **Per-session translation limit** | Maximum translated characters per session |

For FREE plans, a hint "(ανά συνεδρία)" / "(per session)" indicates these reset each session.

---

## Language Setting

The **"Γλώσσα διεπαφής / Interface language"** selector controls the UI language:

| Option | Behavior |
|--------|----------|
| **Αυτόματη (browser)** / **Auto** | Detect language from browser locale (`navigator.language`) |
| **Ελληνικά** | Force Greek UI |
| **English** | Force English UI (with Greek fallback for untranslated strings) |

Changes take effect immediately after saving preferences.

---

## HUD Usage

### Starting/Stopping Captions

1. Navigate to any tab where you want captions
2. The HUD overlay appears (draggable panel)
3. Click **▶ Start** to begin transcription
4. Status changes: **"Σύνδεση…"** / **"Connecting…"** → **"Συνδέθηκε"** / **"Connected"**
5. Captions appear in the transcript area
6. Click **■ Stop** to end the session

### Status Indicators

| Status | Meaning |
|--------|---------|
| **Αναμονή** / **Idle** | Ready to start |
| **Σύνδεση…** / **Connecting…** | Establishing WebSocket connection |
| **Συνδέθηκε** / **Connected** | Session active, transcribing |
| **Τερματισμός…** / **Stopping…** | Closing session |
| **Αποσυνδέθηκε** / **Disconnected** | Session ended |
| **Σφάλμα** / **Error** | Something went wrong |

### Plan & Usage Display

The HUD shows your current plan and usage:

- Plan badge (FREE / PRO / ENTERPRISE)
- Usage summary: minutes and sessions
- Upgrade link for FREE users

### Quality Modes

| Mode | Description |
|------|-------------|
| **FAST** | Lower latency, suitable for real-time conversations |
| **SMART** | Higher quality transcription and translation (PRO/ENTERPRISE only) |

If you're on FREE and select SMART mode:
- Warning: **"⚠️ Το SMART mode δεν είναι διαθέσιμο."**
- Hint: **"Αναβάθμισε για SMART mode."**
- Session falls back to FAST mode

### Auth Required State

If no valid authentication is configured:

- Message: **"Για να χρησιμοποιήσεις το LinguaLive, συνδέσου πρώτα στον λογαριασμό σου."**
- Button: **"Άνοιγμα Ρυθμίσεων"** — Opens the Options page

### Limits Reached State

When you exceed your plan limits:

- Session stops automatically
- Message: **"Έφτασες το όριο του πλάνου σου..."** / **"You've reached your plan limit..."**
- Upgrade CTA: **"Αναβάθμιση"** / **"Upgrade"** — Opens pricing page

---

## Diagnostics

### HUD Error Link

When technical errors occur, the HUD may show:

- **"Διάγνωση προβλήματος"** / **"Diagnose problem"** — Link that opens the Options diagnostics section

### Diagnostics Section in Options

The diagnostics panel helps troubleshoot issues:

#### Quick Check Button

**"🔍 Γρήγορος έλεγχος"** / **"🔍 Quick check"** runs these checks:

| Check | What it tests |
|-------|---------------|
| **Περιβάλλον** / **Environment** | Chrome APIs available, fetch/WebSocket support |
| **Backend** | Server reachability at configured URL |
| **Σύνδεση** / **Connection** | Authentication validity (token check) |
| **Μικρόφωνο** / **Microphone** | Browser microphone permission status |

Results show icons:
- ✅ — OK
- ❌ — Error
- ℹ️ — Info (neutral)
- ⏭️ — Skipped

#### Copy Diagnostics Report

**"📋 Αντιγραφή αναφοράς"** / **"📋 Copy report"** copies a diagnostic report to your clipboard containing:

- Extension version
- Browser information
- Backend URL
- Check results and messages
- Timestamp

This report does NOT include sensitive data like tokens.

---

## Developer Guide

### Commands

Run these from the `extension/` directory:

```bash
# Install dependencies
npm install

# Run ESLint
npm run lint

# Run TypeScript type checking
npm run typecheck

# Build the extension
npm run build
```

### From Repository Root

```bash
# Run all extension checks (mirrors CI)
make check-ext

# Run pre-commit hooks on all files
python -m pre_commit run --all-files
```

### CI Integration

Extension CI is configured in `.github/workflows/ci.yml` and runs:

- `npm run lint`
- `npm run typecheck`
- `npm run build`

### Project Structure

```
extension/
├── dist/                 # Build output
├── docs/                 # Additional documentation
├── src/
│   ├── api/              # Backend client, realtime client
│   ├── audio/            # Microphone recorder
│   ├── background/       # Service worker (tab capture)
│   ├── content/          # HUD content script
│   ├── i18n/             # Internationalization (messages.ts)
│   ├── storage/          # Settings storage
│   └── options.ts        # Options page logic
├── manifest.json         # Chrome extension manifest (MV3)
├── options.html          # Options page HTML
├── package.json          # Node dependencies
└── tsconfig.json         # TypeScript configuration
```

### Additional Documentation

- `extension/docs/extension_dev_guide.md` — Detailed developer setup guide
- `docs/extension-troubleshooting.md` — Troubleshooting common issues (repo root)
