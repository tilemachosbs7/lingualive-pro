const DEFAULT_BACKEND = "http://localhost:8000";

export async function getBackendBaseUrl(): Promise<string> {
  return new Promise((resolve) => {
    chrome.storage.sync.get({ backendUrl: DEFAULT_BACKEND }, (items) => {
      const url = typeof items.backendUrl === "string" && items.backendUrl.trim().length > 0 ? items.backendUrl : DEFAULT_BACKEND;
      resolve(url);
    });
  });
}

export async function translateText(text: string, targetLang: string, sourceLang?: string): Promise<string> {
  const baseUrl = await getBackendBaseUrl();
  const trimmedBase = baseUrl.replace(/\/+$/, "");
  const url = `${trimmedBase}/api/translate-text`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      text,
      target_lang: targetLang,
      source_lang: sourceLang ?? null
    })
  });

  if (!response.ok) {
    const detail = await safeErrorDetail(response);
    throw new Error(detail ?? `Backend error: ${response.status}`);
  }

  const data = (await response.json()) as { translated_text?: string };
  if (!data.translated_text) {
    throw new Error("Backend response missing translation");
  }

  return data.translated_text;
}

export interface TranslateAudioResult {
  originalText: string;
  translatedText: string;
  sourceLang: string;
  targetLang: string;
}

export async function translateAudio(
  audioBlob: Blob,
  opts: { sourceLang: string; targetLang: string }
): Promise<TranslateAudioResult> {
  const baseUrl = await getBackendBaseUrl();
  const trimmedBase = baseUrl.replace(/\/+$/, "");
  const url = `${trimmedBase}/api/translate-audio`;

  const form = new FormData();
  form.append("file", audioBlob, "audio.webm");
  form.append("source_lang", opts.sourceLang);
  form.append("target_lang", opts.targetLang);

  const response = await fetch(url, {
    method: "POST",
    body: form
  });

  if (!response.ok) {
    const detail = await safeErrorDetail(response);
    throw new Error(detail ?? `Backend error: ${response.status}`);
  }

  const data = (await response.json()) as {
    original_text?: string;
    translated_text?: string;
    source_lang?: string;
    target_lang?: string;
  };

  if (!data.original_text || !data.translated_text || !data.target_lang) {
    throw new Error("Backend response missing audio translation data");
  }

  return {
    originalText: data.original_text,
    translatedText: data.translated_text,
    sourceLang: data.source_lang ?? opts.sourceLang,
    targetLang: data.target_lang
  };
}

async function safeErrorDetail(response: Response): Promise<string | null> {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") {
      return data.detail;
    }
  } catch {
    // ignore parse errors
  }
  return null;
}
