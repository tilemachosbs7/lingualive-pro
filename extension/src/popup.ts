const startButton = document.getElementById("lt-start-btn") as HTMLButtonElement | null;
const settingsButton = document.getElementById("lt-settings-btn") as HTMLButtonElement | null;
const popupRoot = document.getElementById("lt-popup-root") as HTMLDivElement | null;
const statusText = document.getElementById("lt-status-text") as HTMLSpanElement | null;
const transcriptionSelect = document.getElementById("lt-transcription-select") as HTMLSelectElement | null;
const translationSelect = document.getElementById("lt-translation-select") as HTMLSelectElement | null;
const transcriptionTip = document.getElementById("lt-transcription-tip") as HTMLParagraphElement | null;
const translationTip = document.getElementById("lt-translation-tip") as HTMLParagraphElement | null;

type PopupHudTheme = 'light' | 'dark';
type TranscriptionProvider = 'deepgram' | 'openai' | 'google';

const TRANSCRIPTION_TIPS: Record<TranscriptionProvider, string> = {
  deepgram: "⚡ Deepgram: Ταχύτερο (300ms), καλύτερο για real-time",
  google: "🔵 Google: Μέτριο (500ms), δωρεάν 60 λεπτά/μήνα",
  openai: "🤖 OpenAI: Αργό (1-2s), πάντα διαθέσιμο"
};

const TRANSLATION_TIPS: Record<string, string> = {
  deepl: "💎 DeepL: Καλύτερη ποιότητα για ευρωπαϊκές γλώσσες",
  openai: "🤖 OpenAI: Καλό για όλες τις γλώσσες"
};

function showHUD(): void {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const activeTab = tabs[0];
    if (!activeTab?.id) return;

    // Στέλνουμε μήνυμα στο content script να ανοίξει το HUD
    chrome.tabs.sendMessage(activeTab.id, {
      type: "SHOW_HUD"
    });
    
    if (statusText) statusText.textContent = "✅ HUD ενεργοποιημένο!";
    window.close(); // Κλείνουμε το popup
  });
}

function openOptions(): void {
  chrome.runtime.openOptionsPage();
}

function applyPopupTheme(theme: PopupHudTheme): void {
  if (!popupRoot) return;
  popupRoot.classList.remove("lt-popup-theme-light", "lt-popup-theme-dark");
  popupRoot.classList.add(theme === 'dark' ? "lt-popup-theme-dark" : "lt-popup-theme-light");
}

function loadPopupTheme(): void {
  chrome.storage.local.get(['languageTranslateHudState'], (result) => {
    const theme = (result?.languageTranslateHudState?.theme as PopupHudTheme) || 'light';
    applyPopupTheme(theme);
  });
}

function loadProviders(): void {
  chrome.storage.sync.get({ 
    transcriptionProvider: 'deepgram',
    translationProvider: 'deepl'
  }, (items) => {
    if (transcriptionSelect) {
      transcriptionSelect.value = items.transcriptionProvider || 'deepgram';
      updateTranscriptionTip(items.transcriptionProvider as TranscriptionProvider);
    }
    if (translationSelect) {
      translationSelect.value = items.translationProvider || 'deepl';
      updateTranslationTip(items.translationProvider);
    }
  });
}

function updateTranscriptionTip(provider: TranscriptionProvider): void {
  if (transcriptionTip) {
    transcriptionTip.textContent = TRANSCRIPTION_TIPS[provider];
  }
}

function updateTranslationTip(provider: string): void {
  if (translationTip) {
    translationTip.textContent = TRANSLATION_TIPS[provider];
  }
}

function saveTranscriptionProvider(provider: TranscriptionProvider): void {
  chrome.storage.sync.set({ transcriptionProvider: provider });
  updateTranscriptionTip(provider);
}

function saveTranslationProvider(provider: string): void {
  chrome.storage.sync.set({ translationProvider: provider });
  updateTranslationTip(provider);
}

// Event listeners
if (startButton) startButton.addEventListener("click", showHUD);
if (settingsButton) settingsButton.addEventListener("click", openOptions);

if (transcriptionSelect) {
  transcriptionSelect.addEventListener("change", (e) => {
    const provider = (e.target as HTMLSelectElement).value as TranscriptionProvider;
    saveTranscriptionProvider(provider);
  });
}

if (translationSelect) {
  translationSelect.addEventListener("change", (e) => {
    const provider = (e.target as HTMLSelectElement).value;
    saveTranslationProvider(provider);
  });
}

loadPopupTheme();
loadProviders();
