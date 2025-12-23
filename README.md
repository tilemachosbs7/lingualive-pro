# 🎙️ LanguageTranslate v2

**Επαγγελματική επέκταση Chrome** για real-time μετάφραση με πολλαπλούς providers αναγνώρισης φωνής και μετάφρασης.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-green.svg)](https://www.google.com/chrome/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

## 📚 Πλήρης Τεκμηρίωση

### 🗺️ [ΠΛΟΗΓΟΣ_ΤΕΚΜΗΡΙΩΣΗΣ.md](ΠΛΟΗΓΟΣ_ΤΕΚΜΗΡΙΩΣΗΣ.md) ← Ξεκίνα εδώ!

**Αναλυτικοί Οδηγοί:**
- 📘 **[ΓΡΗΓΟΡΗ_ΕΚΚΙΝΗΣΗ.md](ΓΡΗΓΟΡΗ_ΕΚΚΙΝΗΣΗ.md)** - Εγκατάσταση σε 4 βήματα (5 λεπτά)
- 📗 **[ΟΔΗΓΟΣ_ΧΡΗΣΗΣ.md](ΟΔΗΓΟΣ_ΧΡΗΣΗΣ.md)** - Πλήρης οδηγός με tips & tricks
- 📕 **[ΤΕΧΝΙΚΗ_ΤΕΚΜΗΡΙΩΣΗ.md](ΤΕΧΝΙΚΗ_ΤΕΚΜΗΡΙΩΣΗ.md)** - Αρχιτεκτονική, API, Data Flow
- 📙 **[PROVIDERS_GUIDE.md](PROVIDERS_GUIDE.md)** - Provider comparison & setup
- 📝 **[ΚΑΤΑΣΤΑΣΗ_ΕΡΓΟΥ.md](ΚΑΤΑΣΤΑΣΗ_ΕΡΓΟΥ.md)** - Implementation status & metrics

---

## 🌟 Βασικά Χαρακτηριστικά

### ⚡ Real-Time Transcription (4 Providers)
- **Deepgram**: ~300ms latency - Το ταχύτερο! (Συνιστώμενο)
- **AssemblyAI**: ~400ms - High accuracy (en/es/fr/de/it/pt only)
- **Google Speech-to-Text**: ~500ms - Δωρεάν 60 λεπτά/μήνα
- **OpenAI Realtime API**: 1-2s - Fallback option

### 🧠 Premium Translation (2 Providers)
- **DeepL**: Κορυφαία ποιότητα για ευρωπαϊκές γλώσσες (Συνιστώμενο)
- **OpenAI GPT-4o-mini**: Καλό για όλες τις γλώσσες

### 🎨 Advanced Features

### 🎨 Advanced HUD v2
- **Dual Theme System**: Light/Dark themes with automatic sync across popup and HUD
- **Fully Customizable Display**:
  - Content font size (12-100px) with slider and +/- buttons
  - UI font size (12-50px) for all interface elements with slider and +/- buttons
  - Text color picker
  - Text background color picker
  - HUD panel background color picker
  - 10+ pre-installed world-class fonts (Arial, Helvetica, Times New Roman, Georgia, Verdana, Courier New, Roboto, Open Sans, Lato, Montserrat)
- **Custom Font Support**:
  - Paste Google Fonts embed link directly
  - Import local font files (.ttf, .otf, .woff, .woff2)
  - Delete custom fonts you no longer need (× button)
  - Custom fonts persist across sessions
- **Drag & Resize**: 
  - Draggable from header with smart boundary clamping
  - Resizable from all 4 corners and 4 edges
  - Minimum size: 260×140px
  - Position and size persist across page reloads
- **Window Controls**:
  - Minimize/Restore functionality
  - Close button
  - Settings gear for display preferences
  - Theme toggle button (🌓)
- **Language Selection**:
  - Source language selector (with Auto-detect option)
  - Target language selector
  - Supports 14 languages: English, Greek, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, Hindi
  - Language preferences persist
- **Reset to Defaults**: One-click button to restore theme defaults for all colors and fonts

### 🔧 Backend API
- FastAPI-based translation service
- OpenAI GPT integration (configurable model)
- Health check endpoint
- CORS-enabled for extension communication
- Environment-based configuration

### 🌐 Chrome Extension
- Manifest V3 compatible
- Content script with persistent HUD overlay
- Background service worker for message routing
- Options page for backend configuration
- Popup with quick demo translation and settings access
- Theme-aware UI across all extension pages

---

## 🚀 Γρήγορη Εκκίνηση

### Για Έλληνες χρήστες:
📖 **Διάβασε τους αναλυτικούς οδηγούς:**
- [📘 ΓΡΗΓΟΡΗ_ΕΚΚΙΝΗΣΗ.md](ΓΡΗΓΟΡΗ_ΕΚΚΙΝΗΣΗ.md) - Εγκατάσταση σε 4 βήματα
- [📗 ΟΔΗΓΟΣ_ΧΡΗΣΗΣ.md](ΟΔΗΓΟΣ_ΧΡΗΣΗΣ.md) - Πλήρης οδηγός με σύγκριση providers

### Quick Start (English):

1. **Start Backend** (One-click):
   ```powershell
   .\start-backend.ps1
   ```

2. **Load Extension**:
   - Open `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked"
   - Select `extension/` folder

3. **Use It**:
   - Click extension icon
   - Choose providers (Deepgram + DeepL recommended)
   - Click "▶️ Ενεργοποίηση HUD"
   - Enjoy real-time translation! 🎉

---

## 📦 Detailed Setup

### Backend Prerequisites
- Python 3.10+
- At least ONE of these API keys:
  - Deepgram API key (recommended - $200 free credit)
  - Google Cloud credentials (60 min/month free)
  - OpenAI API key (fallback)

### Backend Installation

1. **Create virtual environment** (automated in start-backend.ps1):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   cd backend
   pip install -e .
   # or: pip install -r requirements.txt
   ```

3. **Set environment variables**:
   
   Copy the example files and add your API keys:
   ```bash
   cp backend/.env.example backend/.env
   cp backend/google-credentials.example.json backend/google-credentials.json
   ```
   
   Edit the `.env` files with your API keys:
   ```env
   # Required (at least one STT provider)
   OPENAI_API_KEY=your-openai-api-key-here
   
   # Speech-to-Text Providers (pick one or more)
   DEEPGRAM_API_KEY=your-deepgram-api-key-here      # Fastest, recommended
   GOOGLE_CLOUD_CREDENTIALS=./google-credentials.json  # 60 min/month free
   ASSEMBLYAI_API_KEY=your-assemblyai-api-key-here  # High accuracy
   
   # Translation Providers
   DEEPL_API_KEY=your-deepl-api-key-here            # Best quality, recommended
   
   # Optional: AssemblyAI EU endpoint (GDPR compliant)
   # ASSEMBLYAI_STREAMING_BASE_URL=wss://streaming.eu.assemblyai.com/v3/ws
   ```

4. **Run the backend**:
   ```bash
   cd backend
   uvicorn app.main:app --reload --port 8000
   ```

5. **Verify it's running**:
   - Health check: http://localhost:8000/health
   - API docs: http://localhost:8000/docs

## Extension Setup

### Installation

1. **Install dependencies**:
   ```bash
   cd extension
   npm install
   ```

2. **Build the extension**:
   ```bash
   npm run build
   ```
   This creates bundled files in `extension/dist/`

3. **Load in Chrome**:
   - Open `chrome://extensions`
   - Enable **Developer mode** (toggle in top-right)
   - Click **Load unpacked**
   - Select the `extension/` folder (not the `dist/` subfolder)

### Configuration

1. **Set Backend URL**:
   - Click the extension icon → "Open settings"
   - Or right-click extension icon → Options
   - Enter your backend URL (default: `http://localhost:8000`)
   - Click "Save"

2. **Test Translation**:
   - On the Options page, enter text and select target language
   - Click "Μετάφραση" to test the backend connection

## Usage

### Quick Demo
1. Click the extension icon in Chrome toolbar
2. Click "Start demo translation"
3. The HUD will appear on the current page with a demo translation

### HUD Features

#### Display Settings
- Click the **⚙️ gear icon** in HUD header to open display settings
- Adjust:
  - **Font size**: Content text size (12-100px) with slider and +/− buttons
  - **UI font size**: Interface element size (12-50px) with slider and +/− buttons
  - **Font**: Choose from 10 world-class fonts or add custom
  - **Custom font**: Paste Google Fonts `<link>` tag or enter font name
  - **Import font**: Upload local font file (.ttf, .otf, .woff, .woff2)
  - **Text color**: Color of translation text
  - **Text background**: Background color behind text content
  - **HUD background**: Overall panel background color
  - **Delete custom font**: Click × button next to Font dropdown when custom font is selected

#### Window Controls
- **Theme Toggle** (🌓): Switch between light/dark themes
- **Minimize** (−): Collapse HUD to header only, click header to restore
- **Close** (×): Hide HUD completely
- **Drag**: Click and drag from header to reposition
- **Resize**: Drag from any corner or edge to resize (minimum 260×140px)

#### Language Selection
- **From**: Source language (Auto-detect available)
- **To**: Target language
- Settings persist across sessions

#### Reset
- Click **"Reset to theme defaults"** to restore all colors and fonts to default values

### Popup Actions
- **Start demo translation**: Show HUD with Greek → English demo
- **Open settings**: Access extension options page
- Popup theme automatically matches HUD theme

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app initialization
│   │   ├── config.py            # Environment settings
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── translate.py     # Translation endpoint
│   │   └── services/
│   │       ├── __init__.py
│   │       └── translation_service.py  # OpenAI integration
│   ├── pyproject.toml
│   └── requirements.txt
│
├── extension/
│   ├── manifest.json            # Chrome extension manifest (MV3)
│   ├── popup.html               # Extension popup UI
│   ├── options.html             # Options page UI
│   ├── package.json
│   ├── tsconfig.json
│   ├── src/
│   │   ├── background.ts        # Service worker
│   │   ├── popup.ts             # Popup logic
│   │   ├── options.ts           # Options page logic
│   │   ├── api/
│   │   │   └── backendClient.ts # Backend API client
│   │   └── content/
│   │       ├── hud.ts           # HUD overlay UI and logic
│   │       └── hudState.ts      # State management and persistence
│   └── dist/                    # Build output (generated)
│
└── README.md
```

## API Endpoints

### POST /api/translate-text
Translate text using OpenAI.

**Request:**
```json
{
  "text": "Γεια σου κόσμε",
  "source_lang": "el",  // optional, null for auto-detect
  "target_lang": "en"
}
```

**Response:**
```json
{
  "translated_text": "Hello world"
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## Development

### Backend Development
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### Extension Development
```bash
cd extension
npm run build  # Rebuild after changes
```

After building, reload the extension in `chrome://extensions` to see changes.

## Storage & State

### HUD State (chrome.storage.local)
- Window position (top, left)
- Window size (width, height)
- Theme (light/dark)
- Minimized state
- Display preferences (font sizes, colors, font family, panel background)
- Language preferences (source, target)
- Custom fonts list (name, CSS value)

### Extension Settings (chrome.storage.sync)
- Backend URL

## Technologies

- **Backend**: Python 3.10+, FastAPI, Pydantic, httpx, OpenAI API
- **Extension**: TypeScript, Chrome Extension Manifest V3, esbuild
- **Fonts**: Google Fonts API, custom font import with @font-face
- **Storage**: Chrome Storage API (local + sync)

## Troubleshooting

### Backend Issues
- **CORS errors**: Ensure backend is running and CORS_ORIGINS includes `*` or your extension origin
- **API key errors**: Verify `OPENAI_API_KEY` environment variable is set correctly
- **Port conflicts**: Change `BACKEND_PORT` or use `--port` flag with uvicorn

### Extension Issues
- **HUD not appearing**: Check browser console for errors, ensure extension is loaded
- **Backend connection failed**: Verify backend URL in Options, check backend is running
- **Fonts not loading**: Check network tab for Google Fonts requests, verify font name spelling
- **Custom fonts not persisting**: Custom fonts are saved to chrome.storage.local and should persist
- **State not persisting**: Check Chrome storage permissions in manifest.json
- **Theme not syncing**: Reload extension after changes

### Build Issues
- **esbuild errors**: Run `npm install` in extension folder
- **TypeScript errors**: Check `tsconfig.json` and ensure all dependencies are installed

## License

MIT

## Credits

Built with:
- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenAI API](https://openai.com/)
- [Chrome Extensions](https://developer.chrome.com/docs/extensions/)
- [Google Fonts](https://fonts.google.com/)
- [esbuild](https://esbuild.github.io/)
