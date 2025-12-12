/** HUD state model and persistence for LinguaLive v2 */

export type HudTheme = 'light' | 'dark';

export interface HudDisplayPrefs {
  fontSizePx: number; // 14-28 typical
  textColor: string; // CSS color
  backgroundColor: string; // CSS color
  fontFamily: string; // CSS font-family string
  panelBackgroundColor: string; // HUD container background
  uiFontSizePx: number; // general UI font size
}

export interface HudPersistedState {
  top: number;
  left: number;
  width: number;
  height: number;
  theme: HudTheme;
  minimized: boolean;
  displayPrefs: HudDisplayPrefs;
  sourceLang: string;
  targetLang: string;
  customFonts: Array<{ name: string; value: string }>;
}

const STORAGE_KEY = 'languageTranslateHudState';

const DEFAULT_STATE: HudPersistedState = {
  top: -1, // -1 means: compute bottom-right on first load
  left: -1,
  width: 360,
  height: 200,
  theme: 'dark',
  minimized: false,
  displayPrefs: {
    fontSizePx: 18,
    textColor: '#ffffff',
    backgroundColor: '#0d1726',
    fontFamily: 'Arial, Helvetica, sans-serif',
    panelBackgroundColor: '',
    uiFontSizePx: 14,
  },
  sourceLang: 'auto',
  targetLang: 'en',
  customFonts: [],
};

/** Load HUD state from chrome.storage.local */
export async function loadHudState(): Promise<HudPersistedState> {
  return new Promise((resolve) => {
    chrome.storage.local.get([STORAGE_KEY], (result) => {
      const stored = result[STORAGE_KEY] as Partial<HudPersistedState> | undefined;
      const state: HudPersistedState = {
        ...DEFAULT_STATE,
        ...stored,
        displayPrefs: { ...DEFAULT_STATE.displayPrefs, ...(stored?.displayPrefs || {}) },
        customFonts: stored?.customFonts || [],
      };

      // Always compute a centered-right position on load for visibility
      // Center vertically, right side with 40px margin
      state.top = Math.max((window.innerHeight - state.height) / 2, 12);
      state.left = Math.max(window.innerWidth - state.width - 60, 12);

      resolve(state);
    });
  });
}

/** Save partial HUD state to chrome.storage.local */
export async function saveHudState(partial: Partial<HudPersistedState>): Promise<void> {
  return new Promise((resolve) => {
    chrome.storage.local.get([STORAGE_KEY], (result) => {
      const current = (result[STORAGE_KEY] as HudPersistedState) || DEFAULT_STATE;
      const updated: HudPersistedState = {
        ...current,
        ...partial,
        displayPrefs: {
          ...current.displayPrefs,
          ...(partial.displayPrefs || {}),
        },
      };
      chrome.storage.local.set({ [STORAGE_KEY]: updated }, () => resolve());
    });
  });
}
