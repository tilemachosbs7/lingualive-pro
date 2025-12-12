import { translateAudio } from "./api/backendClient";

type LTMessage =
  | { type: "LT_START_TAB_CAPTURE"; payload: { sourceLang: string; targetLang: string } }
  | { type: "LT_STOP_TAB_CAPTURE" }
  | { type: "LT_AUDIO_READY"; payload: { audioBase64: string; sourceLang: string; targetLang: string } };

let currentTabId: number | null = null;

function sendToTab(tabId: number | null, message: unknown): void {
  if (tabId !== null) {
    chrome.tabs.sendMessage(tabId, message).catch(() => {
      // Tab may have been closed
    });
  }
}

async function handleAudioReady(payload: {
  audioBase64: string;
  sourceLang: string;
  targetLang: string;
}): Promise<void> {
  if (currentTabId === null) return;
  const tabId = currentTabId;

  // Notify HUD that we're processing
  sendToTab(tabId, { type: "LT_PROCESSING" });

  try {
    // Convert base64 back to Blob
    const binaryString = atob(payload.audioBase64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    const blob = new Blob([bytes], { type: "audio/webm" });

    // Send to backend for translation
    const result = await translateAudio(blob, {
      sourceLang: payload.sourceLang,
      targetLang: payload.targetLang,
    });

    sendToTab(tabId, { type: "LT_TAB_TRANSLATION_RESULT", payload: result });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Translation failed";
    sendToTab(tabId, { type: "LT_TAB_TRANSLATION_ERROR", payload: { message } });
  }
}

chrome.runtime.onMessage.addListener((message: LTMessage, sender, sendResponse) => {
  if (message?.type === "LT_START_TAB_CAPTURE") {
    // Store the tab ID for later
    currentTabId = sender.tab?.id ?? null;
    sendResponse({ ok: true });
  }
  if (message?.type === "LT_AUDIO_READY") {
    handleAudioReady(message.payload);
    sendResponse({ ok: true });
  }
  return false;
});

console.log("LanguageTranslate background ready");

