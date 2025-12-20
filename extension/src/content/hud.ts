import { loadHudState, saveHudState, HudTheme, HudPersistedState } from './hudState';
import { 
  LANGUAGES, 
  STT_PROVIDERS, 
  TRANSLATION_PROVIDERS,
  getSTTProvidersForLanguage,
  getTranslationProvidersForLanguage,
  searchLanguages,
  LanguageInfo 
} from '../languageData';

const HUD_ID = "lingualive-hud-root";
const MIN_WIDTH = 260;
const MIN_HEIGHT = 140;
const THEME_DEFAULTS: Record<HudTheme, { bg: string; color: string; border: string; btnBg: string }> = {
  light: { bg: 'rgba(248, 250, 252, 0.96)', color: '#0f172a', border: '#d7dce5', btnBg: '#e2e8f0' },
  dark: { bg: 'rgba(11, 19, 32, 0.94)', color: '#e2e8f0', border: '#1f2a44', btnBg: '#1f2a44' },
};

// ============================================================================
// TOAST NOTIFICATION SYSTEM (5.5)
// ============================================================================
type ToastType = 'success' | 'error' | 'warning' | 'info';
interface ToastOptions {
  duration?: number;
  position?: 'top' | 'bottom';
}

const toastQueue: HTMLDivElement[] = [];
const TOAST_CONTAINER_ID = 'lingualive-toast-container';

function createToastContainer(): HTMLDivElement {
  let container = document.getElementById(TOAST_CONTAINER_ID) as HTMLDivElement;
  if (!container) {
    container = document.createElement('div');
    container.id = TOAST_CONTAINER_ID;
    container.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      z-index: 999999;
      display: flex;
      flex-direction: column;
      gap: 10px;
      pointer-events: none;
    `;
    document.body.appendChild(container);
  }
  return container;
}

function showToast(message: string, type: ToastType = 'info', options: ToastOptions = {}): void {
  const { duration = 3000, position = 'top' } = options;
  
  const colors: Record<ToastType, { bg: string; border: string; icon: string }> = {
    success: { bg: '#10b981', border: '#059669', icon: '✓' },
    error: { bg: '#ef4444', border: '#dc2626', icon: '✕' },
    warning: { bg: '#f59e0b', border: '#d97706', icon: '⚠' },
    info: { bg: '#3b82f6', border: '#2563eb', icon: 'ℹ' },
  };
  
  const container = createToastContainer();
  container.style.top = position === 'top' ? '20px' : 'auto';
  container.style.bottom = position === 'bottom' ? '20px' : 'auto';
  
  const toast = document.createElement('div');
  toast.style.cssText = `
    padding: 12px 20px 12px 16px;
    background: ${colors[type].bg};
    border-left: 4px solid ${colors[type].border};
    border-radius: 6px;
    color: white;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    transform: translateX(120%);
    transition: transform 0.3s ease;
    pointer-events: auto;
    max-width: 350px;
  `;
  
  const icon = document.createElement('span');
  icon.textContent = colors[type].icon;
  icon.style.cssText = `font-weight: bold; font-size: 16px;`;
  
  const text = document.createElement('span');
  text.textContent = message;
  
  const closeBtn = document.createElement('button');
  closeBtn.textContent = '×';
  closeBtn.style.cssText = `
    margin-left: auto;
    background: none;
    border: none;
    color: white;
    font-size: 18px;
    cursor: pointer;
    padding: 0 4px;
    opacity: 0.7;
  `;
  closeBtn.onclick = () => removeToast(toast);
  
  toast.appendChild(icon);
  toast.appendChild(text);
  toast.appendChild(closeBtn);
  container.appendChild(toast);
  toastQueue.push(toast);
  
  // Animate in
  requestAnimationFrame(() => {
    toast.style.transform = 'translateX(0)';
  });
  
  // Auto remove
  setTimeout(() => removeToast(toast), duration);
}

function removeToast(toast: HTMLDivElement): void {
  toast.style.transform = 'translateX(120%)';
  setTimeout(() => {
    toast.remove();
    const idx = toastQueue.indexOf(toast);
    if (idx > -1) toastQueue.splice(idx, 1);
  }, 300);
}

// Helper functions for common toasts
function showSuccessToast(msg: string): void { showToast(msg, 'success'); }
function showErrorToast(msg: string): void { showToast(msg, 'error', { duration: 5000 }); }
function showWarningToast(msg: string): void { showToast(msg, 'warning'); }
function showInfoToast(msg: string): void { showToast(msg, 'info'); }

// ============================================================================
// CONFIDENCE VISUALIZATION (5.1)
// ============================================================================
interface ConfidenceInfo {
  level: string;
  color: string;
  percentage: number;
}

function getConfidenceInfo(confidence: number): ConfidenceInfo {
  if (confidence >= 0.9) return { level: 'high', color: '#22c55e', percentage: Math.round(confidence * 100) };
  if (confidence >= 0.7) return { level: 'medium', color: '#eab308', percentage: Math.round(confidence * 100) };
  if (confidence >= 0.5) return { level: 'low', color: '#f97316', percentage: Math.round(confidence * 100) };
  return { level: 'very_low', color: '#ef4444', percentage: Math.round(confidence * 100) };
}

function createConfidenceIndicator(confidence: number): HTMLSpanElement {
  const info = getConfidenceInfo(confidence);
  const indicator = document.createElement('span');
  indicator.style.cssText = `
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: ${info.color};
    margin-left: 6px;
    vertical-align: middle;
  `;
  indicator.title = `Confidence: ${info.percentage}%`;
  return indicator;
}

// ============================================================================
// HUD DEFAULTS AND CONFIGURATION
// ============================================================================

const DEFAULT_DISPLAY_PREFS = {
  fontSizePx: 18,
  textColor: '#ffffff',
  backgroundColor: '#0d1726',
  fontFamily: 'Arial, Helvetica, sans-serif',
  panelBackgroundColor: '',
  uiFontSizePx: 14,
};

const FONT_CHOICES = [
  { value: "Arial, Helvetica, sans-serif", label: "Arial" },
  { value: "Helvetica, Arial, sans-serif", label: "Helvetica" },
  { value: "'Times New Roman', Times, serif", label: "Times New Roman" },
  { value: "Georgia, serif", label: "Georgia" },
  { value: "Verdana, Geneva, sans-serif", label: "Verdana" },
  { value: "'Courier New', Courier, monospace", label: "Courier New" },
  { value: "Roboto, system-ui, sans-serif", label: "Roboto" },
  { value: "'Open Sans', sans-serif", label: "Open Sans" },
  { value: "Lato, sans-serif", label: "Lato" },
  { value: "Montserrat, sans-serif", label: "Montserrat" }
];
type HudMode = "idle" | "capturing" | "processing" | "translated" | "error";
type TranslationProviderType = "deepl" | "openai" | "google";

// References to HUD select elements for smart filtering
let sourceSelectRef: HTMLSelectElement | null = null;
let targetSelectRef: HTMLSelectElement | null = null;
let transcriptionSelectRef: HTMLSelectElement | null = null;
let translationSelectRef: HTMLSelectElement | null = null;
let selectedTranslationProvider: TranslationProviderType = "deepl";

let hudState: HudPersistedState | null = null;
let dragState: { active: boolean; offsetX: number; offsetY: number } = { active: false, offsetX: 0, offsetY: 0 };
let resizeState: { active: boolean; handle: string; startX: number; startY: number; startWidth: number; startHeight: number; startTop: number; startLeft: number } | null = null;
let lastTranslation: { text: string; translatedText: string } | null = null;
let hudMode: HudMode = "idle";
let lastOriginalText = "";
let lastTranslatedText = "";
let lastErrorMessage = "";
let startStopBtn: HTMLButtonElement | null = null;
let originalBox: HTMLDivElement | null = null;
let translatedBox: HTMLDivElement | null = null;
let errorBanner: HTMLDivElement | null = null;

// Audio capture state
let mediaStream: MediaStream | null = null;
let audioContext: AudioContext | null = null;
let audioWorklet: AudioWorkletNode | null = null;
let websocket: WebSocket | null = null;

// Text history for accumulating captions
let originalHistory: string[] = [];
let translatedHistory: string[] = [];

// AAA: Single draft slot - fast pass updates this, refine finalizes to history
let currentDraftTranslation: string | null = null;

// Transcription providers
type TranscriptionProvider = "deepgram" | "openai" | "google" | "assemblyai";
const PROVIDERS: Record<TranscriptionProvider, { endpoint: string; name: string; latency: string; quality: string; sampleRate: number }> = {
  deepgram: { endpoint: "ws://127.0.0.1:8000/api/deepgram", name: "Deepgram", latency: "~300ms", quality: "Excellent", sampleRate: 16000 },
  openai: { endpoint: "ws://127.0.0.1:8000/api/realtime", name: "OpenAI Realtime", latency: "~1-2s", quality: "Good", sampleRate: 24000 },
  google: { endpoint: "ws://127.0.0.1:8000/api/google-speech", name: "Google Cloud", latency: "~500ms", quality: "Good", sampleRate: 16000 },
  assemblyai: { endpoint: "ws://127.0.0.1:8000/api/assemblyai", name: "AssemblyAI", latency: "~400ms", quality: "High", sampleRate: 16000 },
};

let selectedProvider: TranscriptionProvider = "openai";

async function getTranscriptionProvider(): Promise<TranscriptionProvider> {
  return new Promise((resolve) => {
    try {
      chrome.storage.sync.get({ transcriptionProvider: "openai" }, (items) => {
        if (chrome.runtime.lastError) {
          console.error("Storage error:", chrome.runtime.lastError);
          resolve("openai");
          return;
        }
        resolve((items.transcriptionProvider as TranscriptionProvider) || "openai");
      });
    } catch (e) {
      console.error("Failed to get provider:", e);
      resolve("openai");
    }
  });
}

async function getQualityMode(): Promise<"fast" | "quality"> {
  return new Promise((resolve) => {
    try {
      chrome.storage.sync.get({ qualityMode: "fast" }, (items) => {
        if (chrome.runtime.lastError) {
          console.error("Storage error:", chrome.runtime.lastError);
          resolve("fast");
          return;
        }
        const mode = items.qualityMode === "quality" ? "quality" : "fast";
        resolve(mode);
      });
    } catch (e) {
      console.error("Failed to get quality mode:", e);
      resolve("fast");
    }
  });
}

async function setTranscriptionProvider(provider: TranscriptionProvider): Promise<void> {
  return new Promise((resolve) => {
    try {
      chrome.storage.sync.set({ transcriptionProvider: provider }, () => {
        if (chrome.runtime.lastError) {
          console.error("Storage error:", chrome.runtime.lastError);
        }
        resolve();
      });
    } catch (e) {
      console.error("Failed to set provider:", e);
      resolve();
    }
  });
}

async function getSpeechPrefs(): Promise<{ sourceLang: string; targetLang: string }> {
  return new Promise((resolve) => {
    try {
      chrome.storage.sync.get({ speechSourceLang: "auto", speechTargetLang: "en" }, (items) => {
        if (chrome.runtime.lastError) {
          console.error("Storage error:", chrome.runtime.lastError);
          resolve({ sourceLang: "auto", targetLang: "en" });
          return;
        }
        resolve({
          sourceLang: typeof items.speechSourceLang === "string" ? items.speechSourceLang : "auto",
          targetLang: typeof items.speechTargetLang === "string" ? items.speechTargetLang : "en",
        });
      });
    } catch (e) {
      console.error("Failed to get prefs:", e);
      resolve({ sourceLang: "auto", targetLang: "en" });
    }
  });
}

function updateHudUI(): void {
  if (startStopBtn) {
    if (hudMode === "capturing") {
      startStopBtn.textContent = "Stop";
      startStopBtn.disabled = false;
      startStopBtn.style.background = "#ef4444";
      startStopBtn.style.color = "#0b1320";
    } else if (hudMode === "processing") {
      startStopBtn.textContent = "Processing...";
      startStopBtn.disabled = true;
      startStopBtn.style.background = "#cbd5e1";
      startStopBtn.style.color = "#0f172a";
    } else {
      startStopBtn.textContent = "Start captions";
      startStopBtn.disabled = false;
      startStopBtn.style.background = "var(--lt-hud-btn-bg, #1f2a44)";
      startStopBtn.style.color = "var(--lt-hud-color, #e2e8f0)";
    }
  }

  const originalPlaceholder = hudMode === "capturing" ? "Listening..." : "Waiting to start";
  const translatedPlaceholder = hudMode === "processing" ? "Translating..." : "";

  if (originalBox) {
    const historyText = originalHistory.join(" ");
    originalBox.textContent = historyText || originalPlaceholder;
    // Auto-scroll to bottom
    originalBox.scrollTop = originalBox.scrollHeight;
  }
  if (translatedBox) {
    // AAA: Show history + draft (draft in lighter style)
    const historyText = translatedHistory.join(" ");
    
    // If there's a draft, show it with different styling
    if (currentDraftTranslation) {
      // Create a temp container for styled content
      translatedBox.innerHTML = "";
      
      if (historyText) {
        const historySpan = document.createElement("span");
        historySpan.textContent = historyText + " ";
        translatedBox.appendChild(historySpan);
      }
      
      // Draft in italics with slightly lower opacity
      const draftSpan = document.createElement("span");
      draftSpan.textContent = currentDraftTranslation;
      draftSpan.style.fontStyle = "italic";
      draftSpan.style.opacity = "0.75";
      translatedBox.appendChild(draftSpan);
    } else {
      translatedBox.textContent = historyText || translatedPlaceholder;
    }
    // Auto-scroll to bottom
    translatedBox.scrollTop = translatedBox.scrollHeight;
  }

  if (errorBanner) {
    if (hudMode === "error" && lastErrorMessage) {
      errorBanner.textContent = lastErrorMessage;
      errorBanner.style.display = "block";
    } else {
      errorBanner.style.display = "none";
      errorBanner.textContent = "";
    }
  }
}

async function beginCaptions(): Promise<void> {
  hudMode = "capturing";
  lastOriginalText = "";
  lastTranslatedText = "";
  lastErrorMessage = "";
  // Clear history and draft for new session
  originalHistory = [];
  translatedHistory = [];
  currentDraftTranslation = null;  // AAA: Clear draft
  updateHudUI();

  try {
    // Use getDisplayMedia to let user select a tab/screen to capture
    mediaStream = await navigator.mediaDevices.getDisplayMedia({
      video: true, // Required by the API
      audio: true, // Request tab audio
      // @ts-expect-error - systemAudio is a newer API
      systemAudio: "include",
      preferCurrentTab: false,
    });

    // Check if we got audio
    const audioTracks = mediaStream.getAudioTracks();
    if (audioTracks.length === 0) {
      throw new Error("No audio track available. Please select a tab and enable 'Share audio'.");
    }

    // Stop video track since we only need audio
    mediaStream.getVideoTracks().forEach(track => track.stop());

    // Connect to WebSocket for real-time transcription
    await connectWebSocket();

    // Set up audio processing pipeline
    await setupAudioPipeline(mediaStream);

  } catch (err) {
    cleanupCapture();
    hudMode = "error";
    if (err instanceof Error) {
      if (err.name === "NotAllowedError") {
        lastErrorMessage = "Screen sharing was cancelled or denied.";
      } else {
        lastErrorMessage = err.message;
      }
    } else {
      lastErrorMessage = "Failed to capture audio.";
    }
    updateHudUI();
  }
}

async function connectWebSocket(): Promise<void> {
  return new Promise(async (resolve, reject) => {
    const provider = PROVIDERS[selectedProvider];
    const translationProvider = await getTranslationProvider();
    websocket = new WebSocket(provider.endpoint);

    websocket.onopen = async () => {
      console.log("WebSocket connected");
      showSuccessToast(`Connected to ${provider.name}`);
      // Get provider-specific sample rate
      const providerInfo = PROVIDERS[selectedProvider];
      // Get quality mode setting
      const qualityMode = await getQualityMode();
      // Send config
      websocket?.send(JSON.stringify({
        type: "config",
        sourceLang: hudState?.sourceLang ?? "auto",
        targetLang: hudState?.targetLang ?? "en",
        translationProvider: translationProvider,
        sampleRate: providerInfo.sampleRate,
        qualityMode: qualityMode,
      }));
      resolve();
    };

    websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // INSTANT: Streaming partial transcript (word by word)
        if (data.type === "partial") {
          const partialText = String(data.text || "").trim();
          if (partialText && originalBox) {
            // Show streaming text with cursor - continuous flow
            const historyText = originalHistory.length > 0 
              ? originalHistory.join(" ") + " " + partialText + "▊"
              : partialText + "▊";
            originalBox.textContent = historyText;
            originalBox.scrollTop = originalBox.scrollHeight;
          }
        }
        
        // Original transcription complete (before translation)
        else if (data.type === "original_complete") {
          const text = String(data.text || "").trim();
          if (text) {
            originalHistory.push(text);
            lastOriginalText = text;
            updateHudUI();
          }
        }
        
        // Original was corrected (syntax fix)
        else if (data.type === "original_corrected") {
          const original = String(data.original || "");
          const corrected = String(data.corrected || "");
          if (original && corrected && original !== corrected) {
            // Replace the last entry with corrected version
            if (originalHistory.length > 0) {
              originalHistory[originalHistory.length - 1] = corrected;
              lastOriginalText = corrected;
              updateHudUI();
            }
          }
        }
        
        // Translation ready - AAA Studio Single Draft Slot:
        // Fast pass: Update draft (not history) for immediate feedback
        // Refine pass: Finalize to history, clear draft
        else if (data.type === "translation") {
          const isRefine = data.is_refine === true;
          const translation = String(data.translation || "").trim();
          
          if (translation) {
            if (isRefine) {
              // REFINE PASS: Final quality translation
              // Push to history (this is the final version)
              if (lastTranslatedText !== translation) {
                translatedHistory.push(translation);
                lastTranslatedText = translation;
              }
              // Clear draft slot
              currentDraftTranslation = null;
              updateHudUI();
            } else {
              // FAST PASS: Update draft slot only (not history)
              // This provides immediate feedback without filling history
              currentDraftTranslation = translation;
              updateHudUI();
            }
            
            // Show low confidence warning if applicable
            if (data.confidence !== undefined && data.confidence < 0.5) {
              showWarningToast(`Low confidence: ${Math.round(data.confidence * 100)}%`);
            }
          }
        }
        
        // Quality score notification
        else if (data.type === "quality_score") {
          const score = data.score;
          if (score !== undefined && score < 0.6) {
            showWarningToast(`Translation quality: ${Math.round(score * 100)}%`);
          }
        }
        
        else if (data.type === "error") {
          console.error("WebSocket error:", data.message);
          lastErrorMessage = data.message;
          showErrorToast(data.message);
          updateHudUI();
        }
      } catch (e) {
        console.error("Failed to parse WebSocket message:", e);
      }
    };

    websocket.onerror = (error) => {
      console.error("WebSocket error:", error);
      showErrorToast("Connection failed - check if backend is running");
      reject(new Error("WebSocket connection failed"));
    };

    websocket.onclose = () => {
      console.log("WebSocket closed");
      if (hudMode === "capturing") {
        // Unexpected close
        hudMode = "idle";
        updateHudUI();
      }
    };

    // Timeout after 5 seconds
    setTimeout(() => {
      if (websocket?.readyState !== WebSocket.OPEN) {
        reject(new Error("WebSocket connection timeout"));
      }
    }, 5000);
  });
}

async function setupAudioPipeline(stream: MediaStream): Promise<void> {
  // Get the correct sample rate for the selected provider
  const providerInfo = PROVIDERS[selectedProvider];
  const sampleRate = providerInfo.sampleRate;
  
  // Create audio context with provider-specific sample rate
  audioContext = new AudioContext({ sampleRate });
  
  // Create source from stream
  const source = audioContext.createMediaStreamSource(stream);
  
  // Create script processor for raw audio access
  // Note: ScriptProcessorNode is deprecated but AudioWorklet requires more setup
  // Buffer size adjusted based on sample rate for ~100-170ms chunks
  const bufferSize = sampleRate === 16000 ? 2048 : 4096;
  const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
  
  processor.onaudioprocess = (event) => {
    if (!websocket || websocket.readyState !== WebSocket.OPEN) return;
    
    const inputData = event.inputBuffer.getChannelData(0);
    
    // Convert float32 to PCM16
    const pcm16 = new Int16Array(inputData.length);
    for (let i = 0; i < inputData.length; i++) {
      const s = Math.max(-1, Math.min(1, inputData[i]));
      pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    
    // Convert to base64 (for OpenAI/Google/AssemblyAI, which expect base64)
    const uint8 = new Uint8Array(pcm16.buffer);
    let binary = "";
    for (let i = 0; i < uint8.length; i++) {
      binary += String.fromCharCode(uint8[i]);
    }
    const base64 = btoa(binary);
    
    // Send JSON audio
    websocket.send(JSON.stringify({
      type: "audio",
      data: base64,
    }));
  };
  
  // Connect nodes
  source.connect(processor);
  processor.connect(audioContext.destination);
  
  // Mute output (we don't want to hear our own audio)
  const gainNode = audioContext.createGain();
  gainNode.gain.value = 0;
  processor.connect(gainNode);
  gainNode.connect(audioContext.destination);
}

function cleanupCapture(): void {
  // Close WebSocket
  if (websocket) {
    websocket.send(JSON.stringify({ type: "stop" }));
    websocket.close();
    websocket = null;
  }
  
  // Close audio context
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  
  // Stop all tracks
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
  }
}

function stopCaptions(): void {
  cleanupCapture();
  hudMode = "idle";
  updateHudUI();
}

async function createHud(): Promise<HTMLDivElement> {
  const existing = document.getElementById(HUD_ID) as HTMLDivElement | null;
  if (existing) return existing;

  hudState = await loadHudState();

  const root = document.createElement("div");
  root.id = HUD_ID;
  root.className = `lt-hud-theme-${hudState.theme}`;
  root.style.position = "fixed";
  root.style.top = `${hudState.top}px`;
  root.style.left = `${hudState.left}px`;
  root.style.width = `${hudState.width}px`;
  root.style.height = `${hudState.height}px`;
  root.style.zIndex = "2147483647";
  root.style.borderRadius = "12px";
  root.style.boxShadow = "0 8px 24px rgba(0,0,0,0.32)";
  root.style.userSelect = "none";
  root.style.backdropFilter = "blur(6px)";
  root.style.display = "none";  // START HIDDEN - User opens manually
  root.style.flexDirection = "column";
  root.style.overflow = "hidden";

  const header = document.createElement("div");
  header.id = "lt-hud-header";
  header.style.display = "flex";
  header.style.alignItems = "center";
  header.style.justifyContent = "space-between";
  header.style.padding = "8px 10px";
  header.style.cursor = "move";
  header.style.flexShrink = "0";
  header.style.borderBottom = "1px solid var(--lt-hud-border)";

  const title = document.createElement("span");
  title.textContent = "LanguageTranslate";
  title.style.fontWeight = "700";
  title.style.fontSize = "1em";

  const controls = document.createElement("div");
  controls.style.display = "flex";
  controls.style.gap = "6px";

  const settingsBtn = document.createElement("button");
  settingsBtn.setAttribute("aria-label", "Open display settings");
  settingsBtn.title = "Display settings";
  settingsBtn.innerHTML = createGearIcon();
  styleIconButton(settingsBtn);

  const themeBtn = document.createElement("button");
  themeBtn.setAttribute("aria-label", "Change theme");
  themeBtn.title = "Change theme";
  themeBtn.innerHTML = createThemeIcon();
  styleIconButton(themeBtn);
  themeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleTheme(root);
  });

  const minimizeBtn = document.createElement("button");
  minimizeBtn.setAttribute("aria-label", "Minimize panel");
  minimizeBtn.title = "Minimize";
  minimizeBtn.textContent = "−";
  styleIconButton(minimizeBtn);
  minimizeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleMinimize(root);
  });

  const closeBtn = document.createElement("button");
  closeBtn.setAttribute("aria-label", "Close panel");
  closeBtn.title = "Close";
  closeBtn.textContent = "×";
  styleIconButton(closeBtn);
  closeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    closeHud(root);
  });

  controls.appendChild(settingsBtn);
  controls.appendChild(themeBtn);
  controls.appendChild(minimizeBtn);
  controls.appendChild(closeBtn);

  header.appendChild(title);
  header.appendChild(controls);

  const bodyContainer = document.createElement("div");
  bodyContainer.id = "lt-hud-body";
  bodyContainer.style.flex = "1";
  bodyContainer.style.overflow = "auto";
  bodyContainer.style.padding = "10px 12px";
  bodyContainer.style.fontFamily = hudState.displayPrefs.fontFamily;
  bodyContainer.style.lineHeight = "1.4";

  const settingsPanel = createSettingsPanel();
  settingsPanel.style.display = "none";

  // ===== LANGUAGE & PROVIDER SELECTION ROW =====
  const languageRow = document.createElement("div");
  languageRow.style.display = "flex";
  languageRow.style.flexWrap = "wrap";
  languageRow.style.alignItems = "center";
  languageRow.style.gap = "6px";
  languageRow.style.marginBottom = "8px";

  const fromLabel = document.createElement("span");
  fromLabel.textContent = "From";
  fromLabel.style.fontSize = "0.85em";
  fromLabel.style.opacity = "0.8";

  // Source language select with 90+ languages
  const sourceSelect = document.createElement("select");
  sourceSelectRef = sourceSelect;
  styleSelect(sourceSelect);
  populateLanguageSelect(sourceSelect, LANGUAGES, true);
  sourceSelect.value = hudState.sourceLang;

  const toLabel = document.createElement("span");
  toLabel.textContent = "→";
  toLabel.style.fontSize = "0.85em";
  toLabel.style.opacity = "0.8";

  // Target language select with 90+ languages (no auto)
  const targetSelect = document.createElement("select");
  targetSelectRef = targetSelect;
  styleSelect(targetSelect);
  populateLanguageSelect(targetSelect, LANGUAGES.filter(l => l.code !== 'auto'), false);
  targetSelect.value = hudState.targetLang;

  languageRow.appendChild(fromLabel);
  languageRow.appendChild(sourceSelect);
  languageRow.appendChild(toLabel);
  languageRow.appendChild(targetSelect);

  // Provider row (separate for better mobile layout)
  const providerRow = document.createElement("div");
  providerRow.style.display = "flex";
  providerRow.style.alignItems = "center";
  providerRow.style.gap = "6px";
  providerRow.style.marginBottom = "8px";

  const transcriptionLabel = document.createElement("span");
  transcriptionLabel.textContent = "🎙️ STT:";
  transcriptionLabel.style.fontSize = "0.85em";
  transcriptionLabel.style.opacity = "0.8";
  transcriptionLabel.title = "Speech-to-Text provider";

  // STT provider select
  const transcriptionSelect = document.createElement("select");
  transcriptionSelectRef = transcriptionSelect;
  styleSelect(transcriptionSelect);
  Object.entries(STT_PROVIDERS).forEach(([key, info]) => {
    const o = document.createElement("option");
    o.value = key;
    o.textContent = `${info.icon} ${info.name}`;
    transcriptionSelect.appendChild(o);
  });
  transcriptionSelect.value = selectedProvider;
  transcriptionSelect.addEventListener("change", async (e) => {
    selectedProvider = (e.target as HTMLSelectElement).value as TranscriptionProvider;
    await setTranscriptionProvider(selectedProvider);
  });

  const translationLabel = document.createElement("span");
  translationLabel.textContent = "🧠 Trans:";
  translationLabel.style.fontSize = "0.85em";
  translationLabel.style.opacity = "0.8";
  translationLabel.title = "Translation provider";

  // Translation provider select (now with Google!)
  const translationSelect = document.createElement("select");
  translationSelectRef = translationSelect;
  styleSelect(translationSelect);
  Object.entries(TRANSLATION_PROVIDERS).forEach(([key, info]) => {
    const o = document.createElement("option");
    o.value = key;
    o.textContent = `${info.icon} ${info.name}`;
    translationSelect.appendChild(o);
  });
  
  // Load saved translation provider
  getTranslationProvider().then(provider => {
    translationSelect.value = provider;
    selectedTranslationProvider = provider as TranslationProviderType;
  });
  
  translationSelect.addEventListener("change", async (e) => {
    const provider = (e.target as HTMLSelectElement).value;
    selectedTranslationProvider = provider as TranslationProviderType;
    await setTranslationProvider(provider);
  });

  providerRow.appendChild(transcriptionLabel);
  providerRow.appendChild(transcriptionSelect);
  providerRow.appendChild(translationLabel);
  providerRow.appendChild(translationSelect);

  // Warning banner for provider compatibility
  const providerWarning = document.createElement("div");
  providerWarning.id = "lt-provider-warning";
  providerWarning.style.display = "none";
  providerWarning.style.padding = "4px 8px";
  providerWarning.style.marginBottom = "6px";
  providerWarning.style.borderRadius = "6px";
  providerWarning.style.fontSize = "0.8em";
  providerWarning.style.background = "rgba(251, 191, 36, 0.15)";
  providerWarning.style.border = "1px solid rgba(251, 191, 36, 0.3)";
  providerWarning.style.color = "#fbbf24";

  // Function to update provider availability based on language
  const updateProviderAvailability = () => {
    const sourceLang = sourceSelect.value;
    const targetLang = targetSelect.value;
    
    // Check STT provider availability
    const availableSTT = getSTTProvidersForLanguage(sourceLang);
    Array.from(transcriptionSelect.options).forEach(opt => {
      const isAvailable = availableSTT.includes(opt.value as typeof availableSTT[number]);
      opt.disabled = !isAvailable;
      const info = STT_PROVIDERS[opt.value as keyof typeof STT_PROVIDERS];
      opt.textContent = isAvailable 
        ? `${info.icon} ${info.name}` 
        : `${info.icon} ${info.name} ❌`;
    });
    
    // Auto-switch STT if current not available
    if (!availableSTT.includes(selectedProvider) && availableSTT.length > 0) {
      selectedProvider = availableSTT[0] as TranscriptionProvider;
      transcriptionSelect.value = selectedProvider;
      setTranscriptionProvider(selectedProvider);
    }
    
    // Check translation provider availability
    const availableTrans = getTranslationProvidersForLanguage(targetLang);
    Array.from(translationSelect.options).forEach(opt => {
      const isAvailable = availableTrans.includes(opt.value as typeof availableTrans[number]);
      opt.disabled = !isAvailable;
      const info = TRANSLATION_PROVIDERS[opt.value as keyof typeof TRANSLATION_PROVIDERS];
      opt.textContent = isAvailable 
        ? `${info.icon} ${info.name}` 
        : `${info.icon} ${info.name} ❌`;
    });
    
    // Auto-switch translation if current not available  
    if (!availableTrans.includes(selectedTranslationProvider) && availableTrans.length > 0) {
      selectedTranslationProvider = availableTrans[0] as TranslationProviderType;
      translationSelect.value = selectedTranslationProvider;
      setTranslationProvider(selectedTranslationProvider);
    }
    
    // Show warning if any provider not available
    const sttOk = availableSTT.includes(selectedProvider);
    const transOk = availableTrans.includes(selectedTranslationProvider);
    if (!sttOk || !transOk) {
      providerWarning.textContent = !sttOk 
        ? `⚠️ ${STT_PROVIDERS[selectedProvider].name} doesn't support this source language`
        : `⚠️ ${TRANSLATION_PROVIDERS[selectedTranslationProvider].name} doesn't support this target language`;
      providerWarning.style.display = "block";
    } else {
      providerWarning.style.display = "none";
    }
  };
  
  // Add change listeners to update availability
  sourceSelect.addEventListener("change", updateProviderAvailability);
  targetSelect.addEventListener("change", updateProviderAvailability);

  const actionRow = document.createElement("div");
  actionRow.style.display = "flex";
  actionRow.style.alignItems = "center";
  actionRow.style.gap = "8px";
  actionRow.style.marginBottom = "6px";

  startStopBtn = document.createElement("button");
  startStopBtn.textContent = "Start captions";
  styleIconButton(startStopBtn);
  startStopBtn.style.padding = "8px 12px";
  startStopBtn.style.fontWeight = "700";
  startStopBtn.addEventListener("click", () => {
    if (hudMode === "capturing") {
      stopCaptions();
    } else if (hudMode !== "processing") {
      void beginCaptions();
    }
  });

  errorBanner = document.createElement("div");
  errorBanner.style.display = "none";
  errorBanner.style.width = "100%";
  errorBanner.style.border = "1px solid rgba(239,68,68,0.4)";
  errorBanner.style.background = "rgba(239,68,68,0.12)";
  errorBanner.style.color = "#fecdd3";
  errorBanner.style.borderRadius = "8px";
  errorBanner.style.padding = "6px 8px";
  errorBanner.style.fontSize = "0.9em";

  actionRow.appendChild(startStopBtn);

  const label = document.createElement("div");
  label.id = "lt-hud-label";
  label.textContent = "Captions (idle)";
  label.style.opacity = "0.8";
  label.style.fontSize = "0.9em";
  label.style.marginBottom = "6px";

  const refreshLabel = (): void => {
    if (!hudState) return;
    const from = hudState.sourceLang === 'auto' ? 'Auto' : hudState.sourceLang;
    label.textContent = `Captions (${from} → ${hudState.targetLang})`;
  };

  sourceSelect.addEventListener("change", () => {
    if (!hudState) return;
    hudState.sourceLang = sourceSelect.value;
    saveHudState({ sourceLang: hudState.sourceLang });
    refreshLabel();
  });

  targetSelect.addEventListener("change", () => {
    if (!hudState) return;
    hudState.targetLang = targetSelect.value;
    saveHudState({ targetLang: hudState.targetLang });
    refreshLabel();
  });

  const panels = document.createElement("div");
  panels.style.display = "flex";
  panels.style.flexDirection = "row";
  panels.style.gap = "0";
  panels.style.flex = "1";
  panels.style.minHeight = "120px";

  // Create left panel (Original)
  const leftPanel = document.createElement("div");
  leftPanel.style.display = "flex";
  leftPanel.style.flexDirection = "column";
  leftPanel.style.gap = "4px";
  leftPanel.style.flex = "1";
  leftPanel.style.minWidth = "0";
  leftPanel.style.paddingRight = "8px";

  const leftHeading = document.createElement("div");
  leftHeading.textContent = "Original";
  leftHeading.style.fontSize = "0.9em";
  leftHeading.style.opacity = "0.8";
  leftHeading.style.fontWeight = "600";

  originalBox = document.createElement("div");
  originalBox.style.flex = "1";
  originalBox.style.minHeight = "100px";
  originalBox.style.maxHeight = "300px";
  originalBox.style.overflowY = "auto";
  originalBox.style.borderRadius = "10px";
  originalBox.style.padding = "8px";
  originalBox.style.background = "var(--lt-hud-bg, #0b1320)";
  originalBox.style.border = "1px solid var(--lt-hud-border)";
  originalBox.style.whiteSpace = "pre-wrap";
  originalBox.style.wordBreak = "break-word";

  leftPanel.appendChild(leftHeading);
  leftPanel.appendChild(originalBox);

  // Create divider
  const divider = document.createElement("div");
  divider.style.width = "1px";
  divider.style.background = "var(--lt-hud-border, #1f2a44)";
  divider.style.margin = "0 4px";
  divider.style.alignSelf = "stretch";

  // Create right panel (Translation)
  const rightPanel = document.createElement("div");
  rightPanel.style.display = "flex";
  rightPanel.style.flexDirection = "column";
  rightPanel.style.gap = "4px";
  rightPanel.style.flex = "1";
  rightPanel.style.minWidth = "0";
  rightPanel.style.paddingLeft = "8px";

  const rightHeading = document.createElement("div");
  rightHeading.textContent = "Translation";
  rightHeading.style.fontSize = "0.9em";
  rightHeading.style.opacity = "0.8";
  rightHeading.style.fontWeight = "600";

  translatedBox = document.createElement("div");
  translatedBox.style.flex = "1";
  translatedBox.style.minHeight = "100px";
  translatedBox.style.maxHeight = "300px";
  translatedBox.style.overflowY = "auto";
  translatedBox.style.borderRadius = "10px";
  translatedBox.style.padding = "8px";
  translatedBox.style.background = "var(--lt-hud-bg, #0b1320)";
  translatedBox.style.border = "1px solid var(--lt-hud-border)";
  translatedBox.style.whiteSpace = "pre-wrap";
  translatedBox.style.wordBreak = "break-word";

  rightPanel.appendChild(rightHeading);
  rightPanel.appendChild(translatedBox);

  panels.appendChild(leftPanel);
  panels.appendChild(divider);
  panels.appendChild(rightPanel);

  bodyContainer.appendChild(settingsPanel);
  bodyContainer.appendChild(languageRow);
  bodyContainer.appendChild(providerRow);
  bodyContainer.appendChild(providerWarning);
  bodyContainer.appendChild(actionRow);
  bodyContainer.appendChild(errorBanner);
  bodyContainer.appendChild(label);
  bodyContainer.appendChild(panels);

  root.appendChild(header);
  root.appendChild(bodyContainer);

  addResizeHandles(root);
  document.body.appendChild(root);

  applyTheme(root);
  applyDisplayPrefs();
  enableDrag(root, header);
  
  // Initial provider availability check
  updateProviderAvailability();

  settingsBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    settingsPanel.style.display = settingsPanel.style.display === "none" ? "block" : "none";
  });

  header.addEventListener("click", () => {
    if (hudState?.minimized) toggleMinimize(root);
  });

  if (hudState.minimized) {
    bodyContainer.style.display = "none";
    root.style.height = "auto";
  }

  refreshLabel();
  updateHudUI();

  return root;
}

function applyTheme(root: HTMLDivElement): void {
  if (!hudState) return;
  const theme = THEME_DEFAULTS[hudState.theme];
  root.className = `lt-hud-theme-${hudState.theme}`;
  root.style.setProperty('--lt-hud-bg', theme.bg);
  root.style.setProperty('--lt-hud-color', theme.color);
  root.style.setProperty('--lt-hud-border', theme.border);
  root.style.setProperty('--lt-hud-btn-bg', theme.btnBg);
  root.style.background = theme.bg;
  root.style.color = theme.color;
  root.style.border = `1px solid ${theme.border}`;
}

function applyDisplayPrefs(): void {
  if (!hudState) return;
  const prefs = hudState.displayPrefs;
  const currentTheme = hudState.theme;
  const root = document.getElementById(HUD_ID) as HTMLDivElement | null;
  if (root) {
    root.style.fontSize = `${prefs.uiFontSizePx}px`;
  }

  const panels = [originalBox, translatedBox];
  panels.forEach((panel) => {
    if (panel) {
      panel.style.fontSize = `${prefs.fontSizePx}px`;
      panel.style.color = prefs.textColor;
      panel.style.background = prefs.backgroundColor;
      panel.style.fontFamily = prefs.fontFamily;
    }
  });

  const body = document.getElementById("lt-hud-body");
  if (body) {
    (body as HTMLElement).style.fontFamily = prefs.fontFamily;
  }

  if (root) {
    const bg = prefs.panelBackgroundColor?.trim();
    if (bg && bg.length > 0) {
      root.style.background = bg;
    } else {
      root.style.background = THEME_DEFAULTS[currentTheme].bg;
    }
    root.style.borderColor = THEME_DEFAULTS[currentTheme].border;
    root.style.color = THEME_DEFAULTS[currentTheme].color;
  }
}

function styleIconButton(btn: HTMLButtonElement): void {
  btn.style.border = "none";
  btn.style.borderRadius = "6px";
  btn.style.padding = "4px 8px";
  btn.style.cursor = "pointer";
  btn.style.fontSize = "1.15em";
  btn.style.lineHeight = "1";
  btn.style.background = "var(--lt-hud-btn-bg, #1f2a44)";
  btn.style.color = "var(--lt-hud-color, #e2e8f0)";
  btn.style.transition = "opacity 0.15s";
  btn.addEventListener("mouseenter", () => (btn.style.opacity = "0.85"));
  btn.addEventListener("mouseleave", () => (btn.style.opacity = "1"));
}

// Helper to get language flag emoji
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

// Populate language select dropdown
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
    
    const flag = getLanguageFlag(lang.code);
    option.textContent = `${flag} ${lang.name}`;
    select.appendChild(option);
  });
}

function styleSelect(select: HTMLSelectElement): void {
  select.style.padding = "6px 8px";
  select.style.borderRadius = "8px";
  select.style.border = "1px solid var(--lt-hud-border)";
  select.style.background = "var(--lt-hud-bg, #0b1320)";
  select.style.color = "var(--lt-hud-color, #e2e8f0)";
  select.style.fontSize = "0.9em";
}

function enableDrag(root: HTMLDivElement, handle: HTMLElement): void {
  const onMouseMove = (event: MouseEvent): void => {
    if (!dragState.active) return;
    
    let newLeft = event.clientX - dragState.offsetX;
    let newTop = event.clientY - dragState.offsetY;

    const rect = root.getBoundingClientRect();
    const minVisible = 40;
    newLeft = Math.max(-rect.width + minVisible, Math.min(newLeft, window.innerWidth - minVisible));
    newTop = Math.max(-rect.height + minVisible, Math.min(newTop, window.innerHeight - minVisible));

    root.style.left = `${newLeft}px`;
    root.style.top = `${newTop}px`;
  };

  const onMouseUp = (): void => {
    if (!dragState.active) return;
    dragState.active = false;
    document.removeEventListener("mousemove", onMouseMove);
    document.removeEventListener("mouseup", onMouseUp);

    // Save position
    const top = parseFloat(root.style.top);
    const left = parseFloat(root.style.left);
    saveHudState({ top, left });
  };

  handle.addEventListener("mousedown", (event) => {
    const rect = root.getBoundingClientRect();
    dragState = {
      active: true,
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top,
    };
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
    event.preventDefault();
  });
}

function createThemeIcon(): string {
  return `<svg width="18" height="18" viewBox="0 0 18 18" xmlns="http://www.w3.org/2000/svg">
    <circle cx="9" cy="9" r="8" fill="currentColor" opacity="0.18" />
    <path d="M9 1a8 8 0 0 0 0 16V1Z" fill="currentColor" />
  </svg>`;
}

function createGearIcon(): string {
  return `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 9.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5Z" stroke="currentColor" stroke-width="1.6" />
    <path d="M4 12.9c0-.3.02-.6.05-.9l-1.6-1.2 1.6-2.8 1.9.5c.3-.3.7-.6 1.1-.8L7.3 5h3.4l.3 2.5c.4.1.8.3 1.1.5l2-1.1 1.6 2.7-1.7 1.2c.02.2.04.5.04.7 0 .3-.02.6-.05.9l1.6 1.2-1.6 2.8-1.9-.5c-.3.3-.7.6-1.1.8l.4 2.7H7.3l-.3-2.5c-.4-.1-.8-.3-1.1-.5l-2 1.1L2.3 14l1.7-1.2c-.02-.2-.04-.5-.04-.7Z" stroke="currentColor" stroke-width="1.4" />
  </svg>`;
}

function toggleTheme(root: HTMLDivElement): void {
  if (!hudState) return;
  const prevTheme = hudState.theme;
  const nextTheme: HudTheme = hudState.theme === 'dark' ? 'light' : 'dark';
  hudState.theme = nextTheme;

  const prevBg = hudState.displayPrefs.panelBackgroundColor?.trim();
  const prevDefaultBg = THEME_DEFAULTS[prevTheme].bg;
  if (!prevBg || prevBg === prevDefaultBg) {
    hudState.displayPrefs.panelBackgroundColor = '';
  }

  applyTheme(root);
  applyDisplayPrefs();
  saveHudState({ theme: hudState.theme, displayPrefs: hudState.displayPrefs });
}

function toggleMinimize(root: HTMLDivElement): void {
  if (!hudState) return;
  const body = document.getElementById("lt-hud-body") as HTMLElement | null;
  if (!body) return;
  hudState.minimized = !hudState.minimized;

  if (hudState.minimized) {
    body.style.display = "none";
    root.style.height = "auto";
  } else {
    body.style.display = "block";
    root.style.height = `${hudState.height}px`;
  }

  saveHudState({ minimized: hudState.minimized });
}

function closeHud(root: HTMLDivElement): void {
  root.remove();
}

function createSettingsPanel(): HTMLDivElement {
  const panel = document.createElement("div");
  panel.id = "lt-hud-settings";
  panel.style.padding = "8px 10px";
  panel.style.marginBottom = "8px";
  panel.style.border = "1px solid var(--lt-hud-border)";
  panel.style.borderRadius = "10px";
  panel.style.background = "rgba(0,0,0,0.04)";

  const title = document.createElement("div");
  title.textContent = "Display";
  title.style.fontWeight = "600";
  title.style.marginBottom = "6px";
  panel.appendChild(title);

  const row = (labelText: string, input: HTMLElement): HTMLDivElement => {
    const wrapper = document.createElement("div");
    wrapper.style.display = "flex";
    wrapper.style.alignItems = "center";
    wrapper.style.gap = "8px";
    wrapper.style.marginBottom = "6px";
    const label = document.createElement("span");
    label.textContent = labelText;
    label.style.fontSize = "0.9em";
    label.style.minWidth = "90px";
    wrapper.appendChild(label);
    wrapper.appendChild(input);
    return wrapper;
  };

  const fontSizeWrapper = document.createElement("div");
  fontSizeWrapper.style.display = "flex";
  fontSizeWrapper.style.alignItems = "center";
  fontSizeWrapper.style.gap = "4px";

  const fontSizeMinusBtn = document.createElement("button");
  fontSizeMinusBtn.textContent = "−";
  fontSizeMinusBtn.style.padding = "2px 8px";
  fontSizeMinusBtn.style.borderRadius = "6px";
  fontSizeMinusBtn.style.border = "1px solid var(--lt-hud-border)";
  fontSizeMinusBtn.style.background = "var(--lt-hud-btn-bg)";
  fontSizeMinusBtn.style.color = "var(--lt-hud-color)";
  fontSizeMinusBtn.style.cursor = "pointer";
  fontSizeMinusBtn.style.fontSize = "1em";

  const fontSizeInput = document.createElement("input");
  fontSizeInput.type = "range";
  fontSizeInput.min = "12";
  fontSizeInput.max = "100";
  fontSizeInput.value = String(hudState?.displayPrefs.fontSizePx ?? 18);
  fontSizeInput.style.flex = "1";

  const fontSizePlusBtn = document.createElement("button");
  fontSizePlusBtn.textContent = "+";
  fontSizePlusBtn.style.padding = "2px 8px";
  fontSizePlusBtn.style.borderRadius = "6px";
  fontSizePlusBtn.style.border = "1px solid var(--lt-hud-border)";
  fontSizePlusBtn.style.background = "var(--lt-hud-btn-bg)";
  fontSizePlusBtn.style.color = "var(--lt-hud-color)";
  fontSizePlusBtn.style.cursor = "pointer";
  fontSizePlusBtn.style.fontSize = "1em";

  const fontSizeValue = document.createElement("span");
  fontSizeValue.style.fontSize = "0.9em";
  fontSizeValue.style.minWidth = "40px";
  fontSizeValue.style.textAlign = "center";
  fontSizeValue.textContent = `${fontSizeInput.value}px`;

  const updateFontSize = (newSize: number) => {
    if (!hudState) return;
    const clamped = Math.max(12, Math.min(100, newSize));
    hudState.displayPrefs.fontSizePx = clamped;
    fontSizeInput.value = String(clamped);
    fontSizeValue.textContent = `${clamped}px`;
    applyDisplayPrefs();
    saveHudState({ displayPrefs: hudState.displayPrefs });
  };

  fontSizeInput.addEventListener("input", () => {
    updateFontSize(Number(fontSizeInput.value));
  });

  fontSizeMinusBtn.addEventListener("click", () => {
    updateFontSize(Number(fontSizeInput.value) - 1);
  });

  fontSizePlusBtn.addEventListener("click", () => {
    updateFontSize(Number(fontSizeInput.value) + 1);
  });

  fontSizeWrapper.appendChild(fontSizeMinusBtn);
  fontSizeWrapper.appendChild(fontSizeInput);
  fontSizeWrapper.appendChild(fontSizePlusBtn);
  fontSizeWrapper.appendChild(fontSizeValue);

  const uiFontSizeWrapper = document.createElement("div");
  uiFontSizeWrapper.style.display = "flex";
  uiFontSizeWrapper.style.alignItems = "center";
  uiFontSizeWrapper.style.gap = "4px";

  const uiFontSizeMinusBtn = document.createElement("button");
  uiFontSizeMinusBtn.textContent = "−";
  uiFontSizeMinusBtn.style.padding = "2px 8px";
  uiFontSizeMinusBtn.style.borderRadius = "6px";
  uiFontSizeMinusBtn.style.border = "1px solid var(--lt-hud-border)";
  uiFontSizeMinusBtn.style.background = "var(--lt-hud-btn-bg)";
  uiFontSizeMinusBtn.style.color = "var(--lt-hud-color)";
  uiFontSizeMinusBtn.style.cursor = "pointer";
  uiFontSizeMinusBtn.style.fontSize = "1em";

  const uiFontSizeInput = document.createElement("input");
  uiFontSizeInput.type = "range";
  uiFontSizeInput.min = "12";
  uiFontSizeInput.max = "50";
  uiFontSizeInput.value = String(hudState?.displayPrefs.uiFontSizePx ?? DEFAULT_DISPLAY_PREFS.uiFontSizePx);
  uiFontSizeInput.style.flex = "1";

  const uiFontSizePlusBtn = document.createElement("button");
  uiFontSizePlusBtn.textContent = "+";
  uiFontSizePlusBtn.style.padding = "2px 8px";
  uiFontSizePlusBtn.style.borderRadius = "6px";
  uiFontSizePlusBtn.style.border = "1px solid var(--lt-hud-border)";
  uiFontSizePlusBtn.style.background = "var(--lt-hud-btn-bg)";
  uiFontSizePlusBtn.style.color = "var(--lt-hud-color)";
  uiFontSizePlusBtn.style.cursor = "pointer";
  uiFontSizePlusBtn.style.fontSize = "1em";

  const uiFontSizeValue = document.createElement("span");
  uiFontSizeValue.style.fontSize = "0.9em";
  uiFontSizeValue.style.minWidth = "40px";
  uiFontSizeValue.style.textAlign = "center";
  uiFontSizeValue.textContent = `${uiFontSizeInput.value}px`;

  const updateUIFontSize = (newSize: number) => {
    if (!hudState) return;
    const clamped = Math.max(12, Math.min(50, newSize));
    hudState.displayPrefs.uiFontSizePx = clamped;
    uiFontSizeInput.value = String(clamped);
    uiFontSizeValue.textContent = `${clamped}px`;
    applyDisplayPrefs();
    saveHudState({ displayPrefs: hudState.displayPrefs });
  };

  uiFontSizeInput.addEventListener("input", () => {
    updateUIFontSize(Number(uiFontSizeInput.value));
  });

  uiFontSizeMinusBtn.addEventListener("click", () => {
    updateUIFontSize(Number(uiFontSizeInput.value) - 1);
  });

  uiFontSizePlusBtn.addEventListener("click", () => {
    updateUIFontSize(Number(uiFontSizeInput.value) + 1);
  });

  uiFontSizeWrapper.appendChild(uiFontSizeMinusBtn);
  uiFontSizeWrapper.appendChild(uiFontSizeInput);
  uiFontSizeWrapper.appendChild(uiFontSizePlusBtn);
  uiFontSizeWrapper.appendChild(uiFontSizeValue);

  const textColorInput = document.createElement("input");
  textColorInput.type = "color";
  textColorInput.value = hudState?.displayPrefs.textColor ?? "#ffffff";
  textColorInput.addEventListener("input", () => {
    if (!hudState) return;
    hudState.displayPrefs.textColor = textColorInput.value;
    applyDisplayPrefs();
    saveHudState({ displayPrefs: hudState.displayPrefs });
  });

  const bgColorInput = document.createElement("input");
  bgColorInput.type = "color";
  bgColorInput.value = hudState?.displayPrefs.backgroundColor ?? "#0d1726";
  bgColorInput.addEventListener("input", () => {
    if (!hudState) return;
    hudState.displayPrefs.backgroundColor = bgColorInput.value;
    applyDisplayPrefs();
    saveHudState({ displayPrefs: hudState.displayPrefs });
  });

  const panelBgInput = document.createElement("input");
  panelBgInput.type = "color";
  panelBgInput.value = hudState?.displayPrefs.panelBackgroundColor || THEME_DEFAULTS[hudState?.theme ?? 'dark'].bg;
  panelBgInput.addEventListener("input", () => {
    if (!hudState) return;
    hudState.displayPrefs.panelBackgroundColor = panelBgInput.value;
    applyDisplayPrefs();
    saveHudState({ displayPrefs: hudState.displayPrefs });
  });

  const fontSelectWrapper = document.createElement("div");
  fontSelectWrapper.style.display = "flex";
  fontSelectWrapper.style.gap = "4px";
  fontSelectWrapper.style.alignItems = "center";
  fontSelectWrapper.style.flex = "1";

  const fontFamilySelect = document.createElement("select");
  fontFamilySelect.style.flex = "1";
  styleSelect(fontFamilySelect);

  const rebuildFontSelect = () => {
    fontFamilySelect.innerHTML = '';
    
    FONT_CHOICES.forEach(choice => {
      const opt = document.createElement("option");
      opt.value = choice.value;
      opt.textContent = choice.label;
      fontFamilySelect.appendChild(opt);
    });
    
    if (hudState?.customFonts && hudState.customFonts.length > 0) {
      const separator = document.createElement("option");
      separator.disabled = true;
      separator.textContent = "── Custom Fonts ──";
      fontFamilySelect.appendChild(separator);
      
      hudState.customFonts.forEach(cf => {
        const opt = document.createElement("option");
        opt.value = cf.value;
        opt.textContent = cf.name;
        fontFamilySelect.appendChild(opt);
      });
    }
    
    fontFamilySelect.value = hudState?.displayPrefs.fontFamily ?? FONT_CHOICES[0].value;
  };

  rebuildFontSelect();

  const deleteFontBtn = document.createElement("button");
  deleteFontBtn.textContent = "×";
  deleteFontBtn.title = "Delete selected custom font";
  deleteFontBtn.style.padding = "4px 10px";
  deleteFontBtn.style.borderRadius = "6px";
  deleteFontBtn.style.border = "1px solid var(--lt-hud-border)";
  deleteFontBtn.style.background = "var(--lt-hud-btn-bg)";
  deleteFontBtn.style.color = "var(--lt-hud-color)";
  deleteFontBtn.style.cursor = "pointer";
  deleteFontBtn.style.fontSize = "1.2em";
  deleteFontBtn.style.lineHeight = "1";

  deleteFontBtn.addEventListener("click", () => {
    if (!hudState) return;
    const selected = fontFamilySelect.value;
    const isCustom = hudState.customFonts.some(cf => cf.value === selected);
    
    if (isCustom) {
      hudState.customFonts = hudState.customFonts.filter(cf => cf.value !== selected);
      hudState.displayPrefs.fontFamily = FONT_CHOICES[0].value;
      rebuildFontSelect();
      customFontInput.value = FONT_CHOICES[0].value;
      applyDisplayPrefs();
      saveHudState({ displayPrefs: hudState.displayPrefs, customFonts: hudState.customFonts });
    }
  });

  fontSelectWrapper.appendChild(fontFamilySelect);
  fontSelectWrapper.appendChild(deleteFontBtn);
  fontFamilySelect.addEventListener("change", () => {
    if (!hudState) return;
    hudState.displayPrefs.fontFamily = fontFamilySelect.value;
    customFontInput.value = fontFamilySelect.value;
    ensureFontLoaded(fontFamilySelect.value);
    applyDisplayPrefs();
    saveHudState({ displayPrefs: hudState.displayPrefs });
  });

  const customFontWrapper = document.createElement("div");
  customFontWrapper.style.display = "flex";
  customFontWrapper.style.gap = "4px";
  customFontWrapper.style.alignItems = "flex-start";

  const customFontInput = document.createElement("textarea");
  customFontInput.placeholder = "Paste Google Fonts link or font name (e.g. 'Fira Code')";
  customFontInput.style.flex = "1";
  customFontInput.style.padding = "6px 8px";
  customFontInput.style.borderRadius = "8px";
  customFontInput.style.border = "1px solid var(--lt-hud-border)";
  customFontInput.style.background = "var(--lt-hud-bg, #0b1320)";
  customFontInput.style.color = "var(--lt-hud-color, #e2e8f0)";
  customFontInput.style.fontSize = "0.85em";
  customFontInput.style.minHeight = "48px";
  customFontInput.style.resize = "vertical";
  customFontInput.style.fontFamily = "monospace";
  customFontInput.value = hudState?.displayPrefs.fontFamily ?? '';

  const applyFontBtn = document.createElement("button");
  applyFontBtn.textContent = "Apply";
  applyFontBtn.style.padding = "6px 12px";
  applyFontBtn.style.borderRadius = "8px";
  applyFontBtn.style.border = "1px solid var(--lt-hud-border)";
  applyFontBtn.style.background = "var(--lt-hud-btn-bg)";
  applyFontBtn.style.color = "var(--lt-hud-color)";
  applyFontBtn.style.cursor = "pointer";
  applyFontBtn.style.fontSize = "0.85em";
  applyFontBtn.style.whiteSpace = "nowrap";

  const applyCustomFont = () => {
    if (!hudState) return;
    const val = customFontInput.value.trim();
    if (val.length === 0) {
      hudState.displayPrefs.fontFamily = FONT_CHOICES[0].value;
      fontFamilySelect.value = FONT_CHOICES[0].value;
      customFontInput.value = FONT_CHOICES[0].value;
      applyDisplayPrefs();
      saveHudState({ displayPrefs: hudState.displayPrefs });
      return;
    }
    
    let fontFamily = val;
    let linkToInject = '';
    
    if (val.includes('<link') && val.includes('fonts.googleapis.com')) {
      const hrefMatch = val.match(/href=["']([^"']+)["']/);
      if (hrefMatch) {
        linkToInject = hrefMatch[1];
        const familyMatch = linkToInject.match(/family=([^&]+)/);
        if (familyMatch) {
          const families = familyMatch[1].split('&family=');
          const firstFamily = families[0].split(':')[0].replace(/\+/g, ' ');
          fontFamily = `'${decodeURIComponent(firstFamily)}', sans-serif`;
        }
      }
    } else if (val.startsWith('http') && val.includes('fonts.googleapis.com')) {
      linkToInject = val;
      const familyMatch = val.match(/family=([^&]+)/);
      if (familyMatch) {
        const families = familyMatch[1].split('&family=');
        const firstFamily = families[0].split(':')[0].replace(/\+/g, ' ');
        fontFamily = `'${decodeURIComponent(firstFamily)}', sans-serif`;
      }
    } else if (!val.includes(',')) {
      fontFamily = `'${val}', sans-serif`;
    }
    
    hudState.displayPrefs.fontFamily = fontFamily;
    
    const isInDefaults = FONT_CHOICES.some(c => c.value === fontFamily);
    const isInCustom = hudState.customFonts.some(c => c.value === fontFamily);
    
    if (!isInDefaults && !isInCustom) {
      const fontName = fontFamily.replace(/['"]/g, '').split(',')[0].trim();
      hudState.customFonts.push({ name: fontName, value: fontFamily });
      rebuildFontSelect();
    }
    
    fontFamilySelect.value = fontFamily;
    customFontInput.value = fontFamily;
    
    if (linkToInject) {
      const id = 'llv2-custom-font-link';
      const existing = document.getElementById(id);
      if (existing) existing.remove();
      const link = document.createElement('link');
      link.id = id;
      link.rel = 'stylesheet';
      link.href = linkToInject;
      document.head.appendChild(link);
    } else {
      ensureFontLoaded(fontFamily);
    }
    
    applyDisplayPrefs();
    saveHudState({ displayPrefs: hudState.displayPrefs, customFonts: hudState.customFonts });
  };

  applyFontBtn.addEventListener("click", applyCustomFont);
  customFontInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.ctrlKey) {
      applyCustomFont();
    }
  });

  customFontWrapper.appendChild(customFontInput);
  customFontWrapper.appendChild(applyFontBtn);

  const importFontWrapper = document.createElement("div");
  importFontWrapper.style.display = "flex";
  importFontWrapper.style.gap = "4px";
  importFontWrapper.style.alignItems = "center";

  const importFontInput = document.createElement("input");
  importFontInput.type = "file";
  importFontInput.accept = ".ttf,.otf,.woff,.woff2";
  importFontInput.style.flex = "1";
  importFontInput.style.fontSize = "0.85em";
  importFontInput.style.color = "var(--lt-hud-color)";

  const importFontBtn = document.createElement("button");
  importFontBtn.textContent = "Import";
  importFontBtn.style.padding = "6px 12px";
  importFontBtn.style.borderRadius = "8px";
  importFontBtn.style.border = "1px solid var(--lt-hud-border)";
  importFontBtn.style.background = "var(--lt-hud-btn-bg)";
  importFontBtn.style.color = "var(--lt-hud-color)";
  importFontBtn.style.cursor = "pointer";
  importFontBtn.style.fontSize = "0.85em";
  importFontBtn.style.whiteSpace = "nowrap";

  importFontBtn.addEventListener("click", () => {
    const file = importFontInput.files?.[0];
    if (!file || !hudState) return;

    const stateRef = hudState; // Capture reference for callbacks
    const reader = new FileReader();
    reader.onload = (e) => {
      const base64 = e.target?.result as string;
      const fontName = file.name.replace(/\.(ttf|otf|woff2?)$/i, '');
      const format = file.name.match(/\.(ttf|otf|woff2?)$/i)?.[1] || 'truetype';
      const formatMap: Record<string, string> = {
        ttf: 'truetype',
        otf: 'opentype',
        woff: 'woff',
        woff2: 'woff2'
      };

      const styleId = `llv2-imported-font-${fontName.replace(/\s+/g, '-')}`;
      let style = document.getElementById(styleId) as HTMLStyleElement | null;
      if (!style) {
        style = document.createElement('style');
        style.id = styleId;
        document.head.appendChild(style);
      }

      style.textContent = `
        @font-face {
          font-family: '${fontName}';
          src: url('${base64}') format('${formatMap[format.toLowerCase()] || 'truetype'}');
          font-weight: normal;
          font-style: normal;
        }
      `;

      const fontFamily = `'${fontName}', sans-serif`;
      stateRef.displayPrefs.fontFamily = fontFamily;

      const isInCustom = stateRef.customFonts.some(c => c.value === fontFamily);
      if (!isInCustom) {
        stateRef.customFonts.push({ name: fontName, value: fontFamily });
        rebuildFontSelect();
      }

      fontFamilySelect.value = fontFamily;
      customFontInput.value = fontFamily;
      applyDisplayPrefs();
      saveHudState({ displayPrefs: stateRef.displayPrefs, customFonts: stateRef.customFonts });
      importFontInput.value = '';
    };
    reader.readAsDataURL(file);
  });

  importFontWrapper.appendChild(importFontInput);
  importFontWrapper.appendChild(importFontBtn);

  panel.appendChild(row("Font size", fontSizeWrapper));
  panel.appendChild(row("UI font size", uiFontSizeWrapper));
  panel.appendChild(row("Font", fontSelectWrapper));
  panel.appendChild(row("Custom font", customFontWrapper));
  panel.appendChild(row("Import font", importFontWrapper));
  panel.appendChild(row("Text color", textColorInput));
  panel.appendChild(row("Text background", bgColorInput));
  panel.appendChild(row("HUD background", panelBgInput));

  const resetBtn = document.createElement("button");
  resetBtn.textContent = "Reset to theme defaults";
  resetBtn.style.width = "100%";
  resetBtn.style.padding = "8px";
  resetBtn.style.marginTop = "4px";
  resetBtn.style.border = "1px solid var(--lt-hud-border)";
  resetBtn.style.borderRadius = "8px";
  resetBtn.style.background = "var(--lt-hud-btn-bg)";
  resetBtn.style.color = "var(--lt-hud-color)";
  resetBtn.style.cursor = "pointer";
  resetBtn.addEventListener("click", () => {
    if (!hudState) return;
    hudState.displayPrefs = { ...DEFAULT_DISPLAY_PREFS };
    fontSizeInput.value = String(hudState.displayPrefs.fontSizePx);
    fontSizeValue.textContent = `${fontSizeInput.value}px`;
    uiFontSizeInput.value = String(hudState.displayPrefs.uiFontSizePx);
    uiFontSizeValue.textContent = `${uiFontSizeInput.value}px`;
    textColorInput.value = hudState.displayPrefs.textColor;
    bgColorInput.value = hudState.displayPrefs.backgroundColor;
    panelBgInput.value = THEME_DEFAULTS[hudState.theme].bg;
    fontFamilySelect.value = hudState.displayPrefs.fontFamily;
    customFontInput.value = hudState.displayPrefs.fontFamily;
    applyTheme(document.getElementById(HUD_ID) as HTMLDivElement);
    applyDisplayPrefs();
    saveHudState({ displayPrefs: hudState.displayPrefs });
  });

  panel.appendChild(resetBtn);

  return panel;
}

function addResizeHandles(root: HTMLDivElement): void {
  const handles = ['top', 'right', 'bottom', 'left', 'bottom-right', 'bottom-left', 'top-right', 'top-left'];
  
  handles.forEach(pos => {
    const handle = document.createElement("div");
    handle.className = `llv2-resize-handle llv2-resize-${pos}`;
    handle.style.position = "absolute";
    handle.style.zIndex = "10";
    
    if (pos === 'top') {
      handle.style.top = "0";
      handle.style.left = "0";
      handle.style.right = "0";
      handle.style.height = "4px";
      handle.style.cursor = "ns-resize";
    } else if (pos === 'bottom') {
      handle.style.bottom = "0";
      handle.style.left = "0";
      handle.style.right = "0";
      handle.style.height = "4px";
      handle.style.cursor = "ns-resize";
    } else if (pos === 'left') {
      handle.style.top = "0";
      handle.style.bottom = "0";
      handle.style.left = "0";
      handle.style.width = "4px";
      handle.style.cursor = "ew-resize";
    } else if (pos === 'right') {
      handle.style.top = "0";
      handle.style.bottom = "0";
      handle.style.right = "0";
      handle.style.width = "4px";
      handle.style.cursor = "ew-resize";
    } else if (pos === 'bottom-right') {
      handle.style.bottom = "0";
      handle.style.right = "0";
      handle.style.width = "12px";
      handle.style.height = "12px";
      handle.style.cursor = "nwse-resize";
    } else if (pos === 'bottom-left') {
      handle.style.bottom = "0";
      handle.style.left = "0";
      handle.style.width = "12px";
      handle.style.height = "12px";
      handle.style.cursor = "nesw-resize";
    } else if (pos === 'top-right') {
      handle.style.top = "0";
      handle.style.right = "0";
      handle.style.width = "12px";
      handle.style.height = "12px";
      handle.style.cursor = "nesw-resize";
    } else if (pos === 'top-left') {
      handle.style.top = "0";
      handle.style.left = "0";
      handle.style.width = "12px";
      handle.style.height = "12px";
      handle.style.cursor = "nwse-resize";
    }
    
    handle.addEventListener("mousedown", (e) => startResize(e, pos, root));
    root.appendChild(handle);
  });
}

function startResize(event: MouseEvent, handle: string, root: HTMLDivElement): void {
  event.preventDefault();
  event.stopPropagation();
  
  const rect = root.getBoundingClientRect();
  resizeState = {
    active: true,
    handle,
    startX: event.clientX,
    startY: event.clientY,
    startWidth: rect.width,
    startHeight: rect.height,
    startTop: rect.top,
    startLeft: rect.left,
  };
  
  document.addEventListener("mousemove", onResizeMove);
  document.addEventListener("mouseup", onResizeEnd);
}

function onResizeMove(event: MouseEvent): void {
  if (!resizeState?.active || !hudState) return;
  
  const dx = event.clientX - resizeState.startX;
  const dy = event.clientY - resizeState.startY;
  
  const root = document.getElementById(HUD_ID) as HTMLDivElement;
  if (!root) return;
  
  let newWidth = resizeState.startWidth;
  let newHeight = resizeState.startHeight;
  let newTop = resizeState.startTop;
  let newLeft = resizeState.startLeft;
  
  if (resizeState.handle.includes('right')) {
    newWidth = Math.max(MIN_WIDTH, resizeState.startWidth + dx);
  }
  if (resizeState.handle.includes('left')) {
    const potentialWidth = resizeState.startWidth - dx;
    if (potentialWidth >= MIN_WIDTH) {
      newWidth = potentialWidth;
      newLeft = resizeState.startLeft + dx;
    }
  }
  if (resizeState.handle.includes('bottom')) {
    newHeight = Math.max(MIN_HEIGHT, resizeState.startHeight + dy);
  }
  if (resizeState.handle.includes('top')) {
    const potentialHeight = resizeState.startHeight - dy;
    if (potentialHeight >= MIN_HEIGHT) {
      newHeight = potentialHeight;
      newTop = resizeState.startTop + dy;
    }
  }

  // Keep at least part of the HUD visible while resizing
  const minVisible = 40;
  const maxLeft = window.innerWidth - minVisible;
  const maxTop = window.innerHeight - minVisible;
  const minLeft = -newWidth + minVisible;
  const minTop = -newHeight + minVisible;

  newLeft = Math.min(Math.max(newLeft, minLeft), maxLeft);
  newTop = Math.min(Math.max(newTop, minTop), maxTop);

  root.style.width = `${newWidth}px`;
  root.style.height = `${newHeight}px`;
  root.style.top = `${newTop}px`;
  root.style.left = `${newLeft}px`;
}

function onResizeEnd(): void {
  if (!resizeState?.active) return;
  resizeState.active = false;
  
  document.removeEventListener("mousemove", onResizeMove);
  document.removeEventListener("mouseup", onResizeEnd);
  
  const root = document.getElementById(HUD_ID) as HTMLDivElement;
  if (!root || !hudState) return;
  
  const width = parseFloat(root.style.width);
  const height = parseFloat(root.style.height);
  const top = parseFloat(root.style.top);
  const left = parseFloat(root.style.left);
  
  hudState.width = width;
  hudState.height = height;
  hudState.top = top;
  hudState.left = left;
  
  saveHudState({ width, height, top, left });
  resizeState = null;
}

async function updateHud(text: string, translatedText: string): Promise<void> {
  const root = await createHud();
  lastTranslation = { text, translatedText };
  lastOriginalText = text;
  lastTranslatedText = translatedText;
  lastErrorMessage = "";
  hudMode = "translated";
  const label = document.getElementById("lt-hud-label") as HTMLDivElement | null;
  if (label && hudState) {
    const from = hudState.sourceLang === 'auto' ? 'Auto' : hudState.sourceLang;
    label.textContent = `Captions (${from} → ${hudState.targetLang})`;
  }
  updateHudUI();
  root.style.display = "flex";
  root.style.opacity = "1";
}

// Initialize HUD on load
async function initialize(): Promise<void> {
  // Load saved preferences
  hudState = await loadHudState();
  selectedProvider = await getTranscriptionProvider();
  
  await createHud();
}

initialize().catch(console.error);
 
 function ensureFontLoaded(fontFamily: string): void {
   if (!fontFamily) return;
   const normalized = fontFamily.split(',')[0].trim().replace(/['"]/g, '');
   if (!normalized) return;
   const id = `llv2-font-${normalized.replace(/\s+/g, '-')}`;
   if (document.getElementById(id)) return;
   const link = document.createElement('link');
   link.id = id;
   link.rel = 'stylesheet';
   link.href = `https://fonts.googleapis.com/css2?family=${encodeURIComponent(normalized.replace(/\s+/g, '+'))}:wght@400;600;700&display=swap`;
   document.head.appendChild(link);
 }

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "SHOW_HUD") {
    // Εμφάνιση HUD όταν ο χρήστης κάνει κλικ στο popup
    const hudElement = document.getElementById(HUD_ID);
    if (hudElement) {
      hudElement.style.display = "flex";
    }
    sendResponse?.({ ok: true });
    return true;
  }
  if (message?.type === "LLV2_SHOW_DEMO_TRANSLATION") {
    updateHud(String(message.text ?? ""), String(message.translatedText ?? ""));
    sendResponse?.({ ok: true });
  }
  if (message?.type === "LT_PROCESSING") {
    // Audio captured, now processing - don't change mode if we're in continuous capture
    if (hudMode !== "capturing") {
      hudMode = "processing";
      updateHudUI();
    }
  }
  if (message?.type === "LT_TAB_TRANSLATION_RESULT") {
    const newOriginal = String(message.payload?.originalText ?? "").trim();
    const newTranslated = String(message.payload?.translatedText ?? "").trim();
    
    // Append to history if we have new text
    if (newOriginal) {
      originalHistory.push(newOriginal);
      lastOriginalText = newOriginal;
    }
    if (newTranslated) {
      translatedHistory.push(newTranslated);
      lastTranslatedText = newTranslated;
    }
    
    // Stay in capturing mode if stream is still active
    if (!mediaStream || !mediaStream.active) {
      hudMode = "translated";
    }
    lastErrorMessage = "";
    const labelEl = document.getElementById("lt-hud-label") as HTMLDivElement | null;
    if (labelEl) {
      const fromLang = message.payload?.sourceLang ?? hudState?.sourceLang ?? "auto";
      const toLang = message.payload?.targetLang ?? hudState?.targetLang ?? "en";
      labelEl.textContent = `Captions (${fromLang} → ${toLang})`;
    }
    updateHudUI();
  }
  if (message?.type === "LT_TAB_TRANSLATION_ERROR") {
    lastErrorMessage = String(message.payload?.message ?? "Translation failed");
    // Only switch to error mode if we're not actively capturing
    if (!mediaStream || !mediaStream.active) {
      hudMode = "error";
    }
    updateHudUI();
  }
  return false;
});

async function getTranslationProvider(): Promise<string> {
  return new Promise((resolve) => {
    chrome.storage.sync.get({ translationProvider: "deepl" }, (items) => {
      resolve((items.translationProvider as string) || "deepl");
    });
  });
}

async function setTranslationProvider(provider: string): Promise<void> {
  return new Promise((resolve) => {
    chrome.storage.sync.set({ translationProvider: provider }, resolve);
  });
}

