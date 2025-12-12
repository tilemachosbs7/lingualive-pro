import { getBackendBaseUrl, translateText } from "./api/backendClient";

const DEFAULT_BACKEND = "http://localhost:8000";

const backendInput = document.getElementById("backendUrl") as HTMLInputElement | null;
const saveButton = document.getElementById("save-backend") as HTMLButtonElement | null;
const saveStatus = document.getElementById("save-status");
const inputText = document.getElementById("llv2-input-text") as HTMLTextAreaElement | null;
const targetLang = document.getElementById("llv2-target-lang") as HTMLSelectElement | null;
const translateBtn = document.getElementById("llv2-translate-btn") as HTMLButtonElement | null;
const resultEl = document.getElementById("llv2-translate-result");
const errorEl = document.getElementById("llv2-error");
const speechSourceSelect = document.getElementById("lt-speech-source") as HTMLSelectElement | null;
const speechTargetSelect = document.getElementById("lt-speech-target") as HTMLSelectElement | null;
const transcriptionProviderSelect = document.getElementById("lt-transcription-provider") as HTMLSelectElement | null;
const translationProviderSelect = document.getElementById("lt-translation-provider") as HTMLSelectElement | null;

function loadBackendUrl(): void {
  chrome.storage.sync.get({ backendUrl: DEFAULT_BACKEND }, (items) => {
    if (backendInput) {
      backendInput.value = items.backendUrl ?? DEFAULT_BACKEND;
    }
  });
}

function saveBackendUrl(): void {
  if (!backendInput) return;
  const url = backendInput.value.trim() || DEFAULT_BACKEND;
  chrome.storage.sync.set({ backendUrl: url }, () => {
    if (saveStatus) {
      saveStatus.textContent = "Saved backend URL.";
      setTimeout(() => {
        saveStatus.textContent = "";
      }, 1500);
    }
  });
}

function loadSpeechPrefs(): void {
  chrome.storage.sync.get({ speechSourceLang: "auto", speechTargetLang: "en" }, (items) => {
    if (speechSourceSelect) speechSourceSelect.value = items.speechSourceLang ?? "auto";
    if (speechTargetSelect) speechTargetSelect.value = items.speechTargetLang ?? "en";
  });
}

function saveSpeechPrefs(): void {
  const source = speechSourceSelect?.value ?? "auto";
  const target = speechTargetSelect?.value ?? "en";
  chrome.storage.sync.set({ speechSourceLang: source, speechTargetLang: target });
}

async function handleTranslate(): Promise<void> {
  if (!inputText || !targetLang || !translateBtn || !resultEl || !errorEl) return;
  const text = inputText.value.trim();
  const lang = targetLang.value;

  clearMessages();
  if (!text) {
    showError("Παρακαλώ γράψε κείμενο για μετάφραση.");
    return;
  }

  translateBtn.disabled = true;
  translateBtn.textContent = "Translating...";

  try {
    const translated = await translateText(text, lang);
    resultEl.textContent = translated;
  } catch (err) {
    const message = err instanceof Error ? err.message : "Translation failed.";
    showError(message);
  } finally {
    translateBtn.disabled = false;
    translateBtn.textContent = "Μετάφραση";
  }
}

function clearMessages(): void {
  if (resultEl) resultEl.textContent = "";
  if (errorEl) errorEl.textContent = "";
}

function showError(message: string): void {
  if (errorEl) {
    errorEl.textContent = message;
  }
}

function loadTranscriptionProvider(): void {
  chrome.storage.sync.get({ transcriptionProvider: "deepgram" }, (items) => {
    if (transcriptionProviderSelect) {
      transcriptionProviderSelect.value = items.transcriptionProvider ?? "deepgram";
    }
  });
}

function saveTranscriptionProvider(): void {
  const provider = transcriptionProviderSelect?.value ?? "deepgram";
  chrome.storage.sync.set({ transcriptionProvider: provider });
}

function loadTranslationProvider(): void {
  chrome.storage.sync.get({ translationProvider: "deepl" }, (items) => {
    if (translationProviderSelect) {
      translationProviderSelect.value = items.translationProvider ?? "deepl";
    }
  });
}

function saveTranslationProvider(): void {
  const provider = translationProviderSelect?.value ?? "deepl";
  chrome.storage.sync.set({ translationProvider: provider });
}

document.addEventListener("DOMContentLoaded", () => {
  loadBackendUrl();
  loadSpeechPrefs();
  loadTranscriptionProvider();
  loadTranslationProvider();

  if (saveButton) {
    saveButton.addEventListener("click", saveBackendUrl);
  }
  if (translateBtn) {
    translateBtn.addEventListener("click", () => {
      void handleTranslate();
    });
  }

  if (speechSourceSelect) {
    speechSourceSelect.addEventListener("change", saveSpeechPrefs);
  }
  if (speechTargetSelect) {
    speechTargetSelect.addEventListener("change", saveSpeechPrefs);
  }

  if (transcriptionProviderSelect) {
    transcriptionProviderSelect.addEventListener("change", saveTranscriptionProvider);
  }

  if (translationProviderSelect) {
    translationProviderSelect.addEventListener("change", saveTranslationProvider);
  }

  // Show the current backend URL in result for quick visibility.
  void getBackendBaseUrl().then((url) => {
    if (saveStatus) {
      saveStatus.textContent = `Using backend: ${url}`;
      setTimeout(() => {
        if (saveStatus.textContent?.startsWith("Using backend")) {
          saveStatus.textContent = "";
        }
      }, 2000);
    }
  });
});
