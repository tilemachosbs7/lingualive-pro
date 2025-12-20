import { 
  LANGUAGES, 
  STT_PROVIDERS, 
  TRANSLATION_PROVIDERS,
  getSTTProvidersForLanguage,
  getTranslationProvidersForLanguage,
  searchLanguages,
  LanguageInfo
} from './languageData';

// DOM Elements
const startButton = document.getElementById("lt-start-btn") as HTMLButtonElement | null;
const settingsButton = document.getElementById("lt-settings-btn") as HTMLButtonElement | null;
const popupRoot = document.getElementById("lt-popup-root") as HTMLDivElement | null;
const statusText = document.getElementById("lt-status-text") as HTMLSpanElement | null;

const sourceLangSelect = document.getElementById("lt-source-lang") as HTMLSelectElement | null;
const targetLangSelect = document.getElementById("lt-target-lang") as HTMLSelectElement | null;
const sourceSearchInput = document.getElementById("lt-source-search") as HTMLInputElement | null;
const targetSearchInput = document.getElementById("lt-target-search") as HTMLInputElement | null;

const transcriptionSelect = document.getElementById("lt-transcription-select") as HTMLSelectElement | null;
const translationSelect = document.getElementById("lt-translation-select") as HTMLSelectElement | null;
const transcriptionTip = document.getElementById("lt-transcription-tip") as HTMLParagraphElement | null;
const translationTip = document.getElementById("lt-translation-tip") as HTMLParagraphElement | null;

const sttWarning = document.getElementById("lt-stt-warning") as HTMLDivElement | null;
const translationWarning = document.getElementById("lt-translation-warning") as HTMLDivElement | null;

type PopupHudTheme = 'light' | 'dark';
type TranscriptionProvider = 'deepgram' | 'openai' | 'google' | 'assemblyai';
type TranslationProvider = 'deepl' | 'openai' | 'google';

const TRANSCRIPTION_TIPS: Record<TranscriptionProvider, string> = {
  deepgram: "⚡ Deepgram Nova-3: Fastest streaming (~300ms), 31 languages",
  google: "🔵 Google Cloud: Most languages (85+), stable (~500ms)",
  openai: "🤖 OpenAI Realtime: Quality mode (~1-2s), 50 languages",
  assemblyai: "🎯 AssemblyAI: High accuracy (~400ms), 6 languages"
};

const TRANSLATION_TIPS: Record<TranslationProvider, string> = {
  deepl: "💎 DeepL: Best quality for European languages, low-latency mode",
  google: "🌍 Google Cloud: Widest coverage (189 languages)",
  openai: "🤖 OpenAI GPT-4: Quality translation for all languages"
};

// State
let currentSourceLang = 'auto';
let currentTargetLang = 'en';
let currentSTTProvider: TranscriptionProvider = 'deepgram';
let currentTranslationProvider: TranslationProvider = 'deepl';

// Populate language dropdown
function populateLanguageSelect(
  select: HTMLSelectElement, 
  languages: LanguageInfo[], 
  includeAuto: boolean = false
): void {
  select.innerHTML = '';
  
  languages.forEach(lang => {
    if (lang.code === 'auto' && !includeAuto) return;
    
    const option = document.createElement('option');
    option.value = lang.code;
    
    // Add flag emoji based on language
    const flag = getLanguageFlag(lang.code);
    option.textContent = `${flag} ${lang.name} (${lang.nativeName})`;
    select.appendChild(option);
  });
}

function getLanguageFlag(code: string): string {
  const flags: Record<string, string> = {
    'auto': '🌐', 'en': '🇬🇧', 'el': '🇬🇷', 'es': '🇪🇸', 'fr': '🇫🇷',
    'de': '🇩🇪', 'it': '🇮🇹', 'pt': '🇵🇹', 'ru': '🇷🇺', 'zh': '🇨🇳',
    'ja': '🇯🇵', 'ko': '🇰🇷', 'ar': '🇸🇦', 'hi': '🇮🇳', 'nl': '🇳🇱',
    'pl': '🇵🇱', 'tr': '🇹🇷', 'sv': '🇸🇪', 'da': '🇩🇰', 'no': '🇳🇴',
    'fi': '🇫🇮', 'cs': '🇨🇿', 'hu': '🇭🇺', 'ro': '🇷🇴', 'bg': '🇧🇬',
    'uk': '🇺🇦', 'vi': '🇻🇳', 'th': '🇹🇭', 'id': '🇮🇩', 'ms': '🇲🇾',
    'he': '🇮🇱', 'fa': '🇮🇷', 'ur': '🇵🇰', 'bn': '🇧🇩', 'ta': '🇱🇰',
  };
  return flags[code] || '🌍';
}

// Filter languages based on search
function filterLanguages(searchInput: HTMLInputElement, select: HTMLSelectElement, includeAuto: boolean): void {
  const query = searchInput.value;
  const filtered = searchLanguages(query);
  populateLanguageSelect(select, filtered, includeAuto);
  
  // Restore selection if still available
  const currentValue = includeAuto ? currentSourceLang : currentTargetLang;
  if (Array.from(select.options).some(opt => opt.value === currentValue)) {
    select.value = currentValue;
  }
}

// Update provider availability based on selected language
function updateProviderAvailability(): void {
  const sourceLang = currentSourceLang;
  const targetLang = currentTargetLang;
  
  // Get available STT providers for source language
  const availableSTT = getSTTProvidersForLanguage(sourceLang);
  
  // Update STT select options
  if (transcriptionSelect) {
    Array.from(transcriptionSelect.options).forEach(option => {
      const provider = option.value as TranscriptionProvider;
      const isAvailable = availableSTT.includes(provider);
      option.disabled = !isAvailable;
      
      // Update text to show availability
      const providerInfo = STT_PROVIDERS[provider];
      if (!isAvailable) {
        option.textContent = `${providerInfo.icon} ${providerInfo.name} - ❌ Not available for this language`;
      } else {
        option.textContent = `${providerInfo.icon} ${providerInfo.name} (${providerInfo.description}) - ${providerInfo.languageCount} languages`;
      }
    });
    
    // Auto-switch if current provider is not available
    if (!availableSTT.includes(currentSTTProvider)) {
      const firstAvailable = availableSTT[0];
      if (firstAvailable) {
        transcriptionSelect.value = firstAvailable;
        currentSTTProvider = firstAvailable;
        saveSettings();
      }
    }
  }
  
  // Show/hide STT warning
  if (sttWarning) {
    sttWarning.style.display = !availableSTT.includes(currentSTTProvider) ? 'block' : 'none';
  }
  
  // Get available translation providers for target language
  const availableTranslation = getTranslationProvidersForLanguage(targetLang);
  
  // Update translation select options
  if (translationSelect) {
    Array.from(translationSelect.options).forEach(option => {
      const provider = option.value as TranslationProvider;
      const isAvailable = availableTranslation.includes(provider);
      option.disabled = !isAvailable;
      
      const providerInfo = TRANSLATION_PROVIDERS[provider];
      if (!isAvailable) {
        option.textContent = `${providerInfo.icon} ${providerInfo.name} - ❌ Not available for this language`;
      } else {
        option.textContent = `${providerInfo.icon} ${providerInfo.name} (${providerInfo.description}) - ${providerInfo.languageCount} languages`;
      }
    });
    
    // Auto-switch if current provider is not available
    if (!availableTranslation.includes(currentTranslationProvider)) {
      const firstAvailable = availableTranslation[0];
      if (firstAvailable) {
        translationSelect.value = firstAvailable;
        currentTranslationProvider = firstAvailable;
        saveSettings();
      }
    }
  }
  
  // Show/hide translation warning
  if (translationWarning) {
    translationWarning.style.display = !availableTranslation.includes(currentTranslationProvider) ? 'block' : 'none';
  }
}

function showHUD(): void {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const activeTab = tabs[0];
    if (!activeTab?.id) return;

    chrome.tabs.sendMessage(activeTab.id, { type: "SHOW_HUD" });
    
    if (statusText) statusText.textContent = "✅ HUD activated!";
    window.close();
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

function saveSettings(): void {
  chrome.storage.sync.set({
    transcriptionProvider: currentSTTProvider,
    translationProvider: currentTranslationProvider,
    speechSourceLang: currentSourceLang,
    speechTargetLang: currentTargetLang,
  });
}

function loadSettings(): void {
  chrome.storage.sync.get({
    transcriptionProvider: 'deepgram',
    translationProvider: 'deepl',
    speechSourceLang: 'auto',
    speechTargetLang: 'en',
  }, (items) => {
    currentSTTProvider = items.transcriptionProvider as TranscriptionProvider;
    currentTranslationProvider = items.translationProvider as TranslationProvider;
    currentSourceLang = items.speechSourceLang;
    currentTargetLang = items.speechTargetLang;
    
    if (transcriptionSelect) {
      transcriptionSelect.value = currentSTTProvider;
      updateTranscriptionTip(currentSTTProvider);
    }
    if (translationSelect) {
      translationSelect.value = currentTranslationProvider;
      updateTranslationTip(currentTranslationProvider);
    }
    if (sourceLangSelect) {
      sourceLangSelect.value = currentSourceLang;
    }
    if (targetLangSelect) {
      targetLangSelect.value = currentTargetLang;
    }
    
    updateProviderAvailability();
  });
}

function updateTranscriptionTip(provider: TranscriptionProvider): void {
  if (transcriptionTip) {
    transcriptionTip.textContent = TRANSCRIPTION_TIPS[provider];
  }
}

function updateTranslationTip(provider: TranslationProvider): void {
  if (translationTip) {
    translationTip.textContent = TRANSLATION_TIPS[provider];
  }
}

// Initialize
function init(): void {
  // Populate language selects
  if (sourceLangSelect) {
    populateLanguageSelect(sourceLangSelect, LANGUAGES, true);
  }
  if (targetLangSelect) {
    populateLanguageSelect(targetLangSelect, LANGUAGES.filter(l => l.code !== 'auto'), false);
  }
  
  // Event listeners
  if (startButton) startButton.addEventListener("click", showHUD);
  if (settingsButton) settingsButton.addEventListener("click", openOptions);
  
  // Source language search
  if (sourceSearchInput && sourceLangSelect) {
    sourceSearchInput.addEventListener("input", () => {
      filterLanguages(sourceSearchInput, sourceLangSelect, true);
    });
  }
  
  // Target language search
  if (targetSearchInput && targetLangSelect) {
    targetSearchInput.addEventListener("input", () => {
      filterLanguages(targetSearchInput, targetLangSelect, false);
    });
  }
  
  // Source language change
  if (sourceLangSelect) {
    sourceLangSelect.addEventListener("change", (e) => {
      currentSourceLang = (e.target as HTMLSelectElement).value;
      saveSettings();
      updateProviderAvailability();
    });
  }
  
  // Target language change
  if (targetLangSelect) {
    targetLangSelect.addEventListener("change", (e) => {
      currentTargetLang = (e.target as HTMLSelectElement).value;
      saveSettings();
      updateProviderAvailability();
    });
  }
  
  // STT provider change
  if (transcriptionSelect) {
    transcriptionSelect.addEventListener("change", (e) => {
      currentSTTProvider = (e.target as HTMLSelectElement).value as TranscriptionProvider;
      saveSettings();
      updateTranscriptionTip(currentSTTProvider);
    });
  }
  
  // Translation provider change
  if (translationSelect) {
    translationSelect.addEventListener("change", (e) => {
      currentTranslationProvider = (e.target as HTMLSelectElement).value as TranslationProvider;
      saveSettings();
      updateTranslationTip(currentTranslationProvider);
    });
  }
  
  loadPopupTheme();
  loadSettings();
}

init();
