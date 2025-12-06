# LinguaLive Pro Extension – Developer Guide

> **Σκοπός:** Οδηγός για developers που θέλουν να τρέξουν και να δοκιμάσουν το Chrome extension τοπικά.

---

## 1. Overview

Το LinguaLive Pro extension είναι ένα Chrome extension (MV3) που:

- **Injects ένα HUD overlay** στις σελίδες YouTube για real-time captions.
- **Συλλέγει audio** από το μικρόφωνο (Mic) ή από το tab audio (Tab capture).
- **Επικοινωνεί με το backend** μέσω WebSocket (`/ws/captions`) για transcription & translation.
- **Χρησιμοποιεί dev-only auth** (dev tokens ή debug headers) — δεν υπάρχει production auth ακόμα.

### Σχέση με το backend

Το extension συνδέεται στο backend που τρέχει στο `http://localhost:8000`. Το backend παρέχει:

- REST endpoints: `/user/bootstrap`, `/user/limits-summary`, `/dev/tokens`
- WebSocket endpoint: `/ws/captions` για real-time audio streaming και transcription

Για περισσότερες λεπτομέρειες, δες το **backend API reference** στο `backend/docs/backend_m2_api_reference.md`.

---

## 2. Prerequisites

Πριν ξεκινήσεις, βεβαιώσου ότι έχεις:

| Απαίτηση | Λεπτομέρειες |
|----------|--------------|
| **Node.js + npm** | Έκδοση 18+ συνιστάται |
| **Python backend** | Τρέχει στο `http://localhost:8000` (δες backend docs) |
| **Chrome / Chromium** | Με Developer Mode enabled |

---

## 3. Install & Build

```bash
cd extension
npm install
npm run typecheck
npm run build
```

### Τι παράγει το build

Μετά το `npm run build`, ο φάκελος `dist/` περιέχει:

| Αρχείο | Περιγραφή |
|--------|-----------|
| `background.js` | Background service worker (tab capture) |
| `content/hud.js` | Content script (HUD overlay) |
| `options.js` | Options page script |
| `*.d.ts` | TypeScript declarations |
| `*.js.map` | Sourcemaps για debugging |

---

## 4. Loading the Extension in Chrome (Dev Mode)

### Βήμα-βήμα

1. Άνοιξε το Chrome και πήγαινε στο: `chrome://extensions`

2. Ενεργοποίησε το **Developer mode** (toggle πάνω δεξιά)

3. Κάνε κλικ στο **"Load unpacked"**

4. Επίλεξε τον φάκελο `extension/` (όχι το `dist/`, αλλά το root folder του extension)

5. Επιβεβαίωσε ότι εμφανίζεται το **"LinguaLive Pro (Dev)"**

### Verification checklist

- ✅ Extension εμφανίζεται στη λίστα
- ✅ Background / Service Worker: "Active" ή "Inspect" link διαθέσιμο
- ✅ Permissions: `storage`, `tabCapture`
- ✅ Host permissions: `http://localhost:8000/*`

---

## 5. Configuring Backend & Auth in Options

### Πώς να ανοίξεις τα Options

- Κάνε κλικ στο extension icon → "Details" → "Extension options"
- Ή απευθείας: `chrome-extension://<extension-id>/options.html`

### Options Page Sections

#### Backend & Auth

| Πεδίο | Περιγραφή |
|-------|-----------|
| **Backend base URL** | Συνήθως `http://localhost:8000` σε dev |
| **Dev token (x-dev-token)** | Token από το `/dev/tokens` endpoint — έχει προτεραιότητα |
| **Debug user id** | Χρησιμοποιείται αν δεν υπάρχει dev token (header: `x-debug-user-id`) |
| **Debug plan** | `FREE`, `PRO`, ή `ENTERPRISE` (header: `x-debug-plan`) |

> **Σημείωση:** Αν έχεις dev token, τα debug headers αγνοούνται. Δες το backend API reference για το πώς να δημιουργήσεις dev tokens.

#### Preferences

| Πεδίο | Περιγραφή |
|-------|-----------|
| **Source language** | BCP-47 code, π.χ. `en`, `el`, `es` |
| **Target language** | BCP-47 code για translation |
| **Quality mode** | `FAST` (χαμηλή καθυστέρηση) ή `SMART` (καλύτερη ποιότητα) |

> **Προσοχή:** Το `SMART` mode δεν επιτρέπεται για FREE plan — θα πάρεις `QUALITY_NOT_ALLOWED_FOR_PLAN` error.

#### Account / Limits / Usage & Billing

Αυτές οι sections εμφανίζουν δεδομένα από τα endpoints:

- `GET /user/bootstrap`
- `GET /user/limits-summary`

Πάτα **"Refresh"** για να ανανεώσεις τα δεδομένα.

---

## 6. Using the HUD on YouTube

### Αυτόματο injection

Το HUD εμφανίζεται αυτόματα όταν επισκέπτεσαι μια σελίδα YouTube (`*://*.youtube.com/*`).

Εμφανίζεται στην κάτω δεξιά γωνία της σελίδας.

### HUD Controls

| Στοιχείο | Περιγραφή |
|----------|-----------|
| **Title** | "LinguaLive Pro" |
| **Session config** | `sourceLang → targetLang · qualityMode` |
| **Status** | `Idle` / `Connecting…` / `Session started (mic on)` / `Error` |
| **Audio source selector** | Radio buttons: **Mic** ή **Tab audio** |
| **▶ Start** | Ξεκινά το session |
| **■ Stop** | Σταματά το session |
| **Transcript area** | Εμφανίζει partial + final captions |
| **✕ Close** | Κλείνει το HUD |

### Audio Source: Mic vs Tab

| Mode | Πώς λειτουργεί |
|------|----------------|
| **Mic** | Χρησιμοποιεί `getUserMedia` για να πάρει audio από το μικρόφωνο του browser. Θα ζητηθεί permission. |
| **Tab audio** | Χρησιμοποιεί `chrome.tabCapture` στο background service worker για να καταγράψει τον ήχο του τρέχοντος tab (π.χ. το video του YouTube). |

### Τι συμβαίνει όταν πατάς Start

1. Φορτώνονται τα settings (backend URL, auth, languages, quality)
2. Δημιουργείται WebSocket connection στο `/ws/captions`
3. Στέλνεται `session_start` envelope με source/target lang
4. Ξεκινά audio capture (Mic ή Tab)
5. Στέλνονται `audio_chunk` envelopes κάθε ~250ms
6. Το backend απαντά με `transcript` envelopes (partial + final, μέσω του `is_final` flag)
7. Τα transcripts εμφανίζονται στο HUD

### Keyboard shortcut

- **Escape**: Κλείνει το HUD (και σταματά το session αν τρέχει)

---

## 7. Common Error Cases & Debug Tips

| Symptom | Likely Cause | How to Debug |
|---------|--------------|--------------|
| **"No auth configured"** στο HUD | Λείπει dev token και debug headers | Συμπλήρωσε το Backend & Auth section στα Options |
| **Network error / cannot connect** | Backend δεν τρέχει ή λάθος URL | Έλεγξε backend logs και το Backend base URL |
| **QUALITY_NOT_ALLOWED_FOR_PLAN** | FREE plan με SMART quality | Άλλαξε σε FAST ή χρησιμοποίησε PRO/ENTERPRISE plan |
| **No HUD on YouTube** | Extension not loaded ή λάθος URL pattern | Έλεγξε `chrome://extensions` και reload extension |
| **Mic permission denied** | Ο χρήστης αρνήθηκε mic access | Click "Allow" στο browser prompt ή reset permissions |
| **Tab capture failed** | Background service worker issue | Inspect service worker από `chrome://extensions` |

### Debugging Tools

1. **Chrome DevTools (F12)** στο YouTube tab:
   - **Console**: Errors από το HUD content script
   - **Network → WS**: WebSocket frames

2. **Service Worker DevTools**:
   - Στο `chrome://extensions` → "LinguaLive Pro (Dev)" → "Service worker" link
   - Console logs από το background script

3. **Extension Errors**:
   - `chrome://extensions` → "Errors" link αν υπάρχουν

---

## 8. Commands Recap

```bash
# Build extension
cd extension
npm run typecheck
npm run build

# Run backend tests (from project root)
cd ..
python -m pytest backend/tests -v --tb=short
```

### Useful npm scripts

| Script | Περιγραφή |
|--------|-----------|
| `npm run typecheck` | TypeScript type checking χωρίς emit |
| `npm run build` | Compile TypeScript → `dist/` |
| `npm run clean` | (αν υπάρχει) Καθαρισμός dist folder |

---

## 9. File Structure Overview

```
extension/
├── manifest.json          # MV3 manifest
├── options.html           # Options page HTML
├── src/
│   ├── storage/
│   │   └── settings.ts    # LinguaLiveSettings, loadSettings, saveSettings
│   ├── api/
│   │   ├── backendClient.ts   # HTTP client for REST endpoints
│   │   └── realtimeClient.ts  # WebSocket client for /ws/captions
│   ├── audio/
│   │   └── micRecorder.ts     # Mic capture with getUserMedia
│   ├── content/
│   │   └── hud.ts             # HUD overlay content script
│   ├── background.ts          # Background service worker (tab capture)
│   └── options.ts             # Options page script
├── dist/                  # Compiled output (after build)
└── docs/
    └── extension_dev_guide.md  # This file
```

---

## 10. Next Steps

Αυτό το extension είναι **dev-only**. Σε production θα χρειαστούν:

- [ ] Real OAuth/JWT auth αντί για dev tokens
- [ ] Proper error handling & retry logic
- [ ] Production ASR/translation services αντί για mock
- [ ] Extension signing & Chrome Web Store deployment

Για περισσότερες λεπτομέρειες, δες τα backend docs και το project roadmap.

---

## 11. Chrome Web Store Localization (`_locales`)

Το Chrome Web Store υποστηρίζει **πολύγλωσσα listings** μέσω του φακέλου `_locales`. Αυτό επιτρέπει στο extension να εμφανίζει διαφορετικό όνομα και περιγραφή ανάλογα με τη γλώσσα του browser του χρήστη.

### Πώς λειτουργεί

1. Το `manifest.json` χρησιμοποιεί **message placeholders** αντί για hardcoded strings:
   ```json
   {
     "name": "__MSG_extName__",
     "description": "__MSG_extDescription__",
     "default_locale": "en"
   }
   ```

2. Το Chrome αντικαθιστά τα `__MSG_*__` με τα αντίστοιχα strings από το `_locales/<locale>/messages.json`.

3. Αν η γλώσσα του χρήστη δεν υπάρχει στα `_locales`, χρησιμοποιείται το `default_locale` (στην περίπτωσή μας, `en`).

### Δομή αρχείων

```
extension/
├── _locales/
│   ├── en/
│   │   └── messages.json   # English (default)
│   └── el/
│       └── messages.json   # Greek
└── manifest.json
```

### Μορφή `messages.json`

Κάθε αρχείο `messages.json` περιέχει key-value pairs:

```json
{
  "extName": {
    "message": "LinguaLive Pro – Live Captions & Translation"
  },
  "extDescription": {
    "message": "Real-time captions and translation overlay for any tab, powered by LinguaLive Pro."
  }
}
```

- **Key** (π.χ. `extName`): Αναφέρεται στο manifest ως `__MSG_extName__`
- **message**: Το πραγματικό string που θα εμφανιστεί

### Πώς να προσθέσεις νέα γλώσσα

| Βήμα | Ενέργεια |
|------|----------|
| 1 | Δημιούργησε φάκελο `extension/_locales/<locale>/` (π.χ. `es` για Ισπανικά) |
| 2 | Δημιούργησε `messages.json` με μεταφρασμένα `extName` και `extDescription` |
| 3 | Build & reload extension — δοκίμασε με Chrome σε διαφορετική γλώσσα |

> **Σημείωση:** Τα BCP-47 locale codes πρέπει να είναι σε lowercase (π.χ. `en`, `el`, `es`, `de`).

### Σχέση με το internal i18n

| Σύστημα | Σκοπός | Αρχεία |
|---------|--------|--------|
| **`_locales`** | Chrome Web Store listing (name, description) | `_locales/*/messages.json` |
| **Internal i18n** | UI strings στο HUD & Options page | `src/i18n/messages.ts` |

Τα δύο συστήματα είναι **ανεξάρτητα** — το `_locales` είναι μόνο για το Chrome API, ενώ το internal i18n είναι για το UI του extension.

---

*Τελευταία ενημέρωση: Δεκέμβριος 2025*
