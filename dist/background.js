/**
 * =============================================================================
 * BACKGROUND.TS — Background Service Worker for LinguaLive Pro
 * =============================================================================
 *
 * Τι κάνει αυτό το module;
 * ------------------------
 * Υλοποιεί tab audio capture χρησιμοποιώντας chrome.tabCapture API.
 * Αυτό είναι dev-only tab audio capture για το τρέχον active tab όπου είναι ο HUD.
 *
 * Ακούει μηνύματα από το HUD content script:
 * - LLP_START_TAB_CAPTURE: ξεκινά capture από το current tab
 * - LLP_STOP_TAB_CAPTURE: σταματά το capture
 *
 * Στέλνει audio chunks πίσω στο content script ως:
 * - LLP_TAB_AUDIO_CHUNK: ArrayBuffer + timestampMs
 *
 * ΣΗΜΕΙΩΣΕΙΣ:
 * -----------
 * - Χρησιμοποιεί MediaRecorder για 250ms slices.
 * - Απαιτεί "tabCapture" permission στο manifest.
 * - Σε production, μπορεί να χρειαστεί πιο robust error handling.
 * =============================================================================
 */
// =============================================================================
// STATE
// =============================================================================
let currentStream = null;
let currentRecorder = null;
let currentTabId = null;
let captureStartTime = 0;
// =============================================================================
// HANDLERS
// =============================================================================
/**
 * Start capturing audio from the sender's tab.
 * Χρησιμοποιεί chrome.tabCapture.capture για να πάρει audio stream.
 */
async function handleStartTabCapture(sender, sendResponse) {
    // Validate sender tab
    const tabId = sender.tab?.id;
    if (typeof tabId !== "number") {
        sendResponse({ ok: false, error: "NO_TAB" });
        return;
    }
    // If already capturing, stop first
    if (currentRecorder || currentStream) {
        handleStopTabCapture();
    }
    try {
        // Request tab audio capture
        // Σημείωση: chrome.tabCapture.capture πρέπει να καλείται από background
        const stream = await new Promise((resolve, reject) => {
            chrome.tabCapture.capture({ audio: true, video: false }, (capturedStream) => {
                if (chrome.runtime.lastError) {
                    reject(new Error(chrome.runtime.lastError.message));
                    return;
                }
                if (!capturedStream) {
                    reject(new Error("No stream returned"));
                    return;
                }
                resolve(capturedStream);
            });
        });
        currentTabId = tabId;
        currentStream = stream;
        captureStartTime = performance.now();
        // Create MediaRecorder
        const options = {
            mimeType: "audio/webm;codecs=opus",
        };
        const recorder = new MediaRecorder(stream, options);
        currentRecorder = recorder;
        // Handle data available
        recorder.ondataavailable = async (event) => {
            if (event.data.size === 0)
                return;
            if (currentTabId === null)
                return;
            try {
                const buffer = await event.data.arrayBuffer();
                const timestampMs = performance.now() - captureStartTime;
                // Send chunk to content script
                const message = {
                    type: "LLP_TAB_AUDIO_CHUNK",
                    buffer,
                    timestampMs,
                };
                chrome.tabs.sendMessage(currentTabId, message).catch((err) => {
                    console.warn("[background] Failed to send audio chunk to tab", err);
                });
            }
            catch (err) {
                console.error("[background] Error processing audio chunk", err);
            }
        };
        // Handle errors
        recorder.onerror = (event) => {
            console.error("[background] MediaRecorder error", event);
            handleStopTabCapture();
        };
        // Handle stop
        recorder.onstop = () => {
            // Cleanup stream tracks when recorder stops
            if (currentStream) {
                currentStream.getTracks().forEach((track) => track.stop());
                currentStream = null;
            }
        };
        // Start recording with 250ms slices
        recorder.start(250);
        console.log("[background] Tab capture started for tab", tabId);
        sendResponse({ ok: true });
    }
    catch (err) {
        console.error("[background] Failed to start tab capture", err);
        sendResponse({
            ok: false,
            error: err instanceof Error ? err.message : "CAPTURE_FAILED",
        });
    }
}
/**
 * Stop the current tab capture.
 * Καθαρίζει recorder + stream + state.
 */
function handleStopTabCapture() {
    // Stop recorder if active
    if (currentRecorder && currentRecorder.state !== "inactive") {
        try {
            currentRecorder.stop();
        }
        catch {
            // Ignore errors during stop
        }
    }
    currentRecorder = null;
    // Stop stream tracks
    if (currentStream) {
        currentStream.getTracks().forEach((track) => {
            try {
                track.stop();
            }
            catch {
                // Ignore errors
            }
        });
        currentStream = null;
    }
    // Reset state
    currentTabId = null;
    captureStartTime = 0;
    console.log("[background] Tab capture stopped");
}
// =============================================================================
// MESSAGE LISTENER
// =============================================================================
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "LLP_START_TAB_CAPTURE") {
        // Async handler - return true to keep sendResponse alive
        void handleStartTabCapture(sender, sendResponse);
        return true;
    }
    if (message.type === "LLP_STOP_TAB_CAPTURE") {
        handleStopTabCapture();
        sendResponse({ ok: true });
        return false;
    }
    // Unknown message - don't respond
    return false;
});
console.log("[background] LinguaLive Pro background service worker loaded");
// =============================================================================
// KEYBOARD SHORTCUTS (chrome.commands)
// =============================================================================
/**
 * Handle keyboard shortcuts from chrome.commands.
 * - "toggle-captions": Send toggle message to active tab's content script
 * - "open-options": Open the extension's options page
 */
chrome.commands.onCommand.addListener(async (command) => {
    console.log("[background] Received command:", command);
    if (command === "open-options") {
        chrome.runtime.openOptionsPage();
        return;
    }
    if (command === "toggle-captions") {
        // Query the active tab in the current window
        const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
        const activeTab = tabs[0];
        if (!activeTab?.id) {
            console.warn("[background] No active tab found for toggle-captions");
            return;
        }
        // Send toggle message to the content script
        try {
            await chrome.tabs.sendMessage(activeTab.id, { type: "LLP_TOGGLE_CAPTIONS" });
        }
        catch (err) {
            // Content script may not be loaded yet on some tabs (e.g., chrome:// pages)
            console.warn("[background] Could not send toggle to tab", activeTab.id, err);
        }
        return;
    }
});
// =============================================================================
// CONTEXT MENU
// =============================================================================
/**
 * Create context menu item on extension install/update.
 * Allows users to right-click and toggle captions.
 */
chrome.runtime.onInstalled.addListener(() => {
    try {
        chrome.contextMenus.create({
            id: "lingualive-toggle-captions",
            title: chrome.i18n.getMessage("contextMenuToggleCaptions") || "Toggle LinguaLive captions",
            contexts: ["page", "video", "audio"],
        });
        console.log("[background] Context menu created");
    }
    catch (err) {
        // Log but don't crash the worker
        console.warn("[background] Failed to create LinguaLive context menu", err);
    }
});
/**
 * Handle context menu clicks.
 * Sends LLP_TOGGLE_CAPTIONS to the tab where the menu was invoked.
 */
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
    if (info.menuItemId !== "lingualive-toggle-captions") {
        return;
    }
    if (!tab || tab.id == null) {
        console.warn("[background] No tab available for context menu toggle");
        return;
    }
    try {
        await chrome.tabs.sendMessage(tab.id, { type: "LLP_TOGGLE_CAPTIONS" });
    }
    catch (err) {
        // Content script may not be loaded on this tab
        console.warn("[background] Failed to send LLP_TOGGLE_CAPTIONS from context menu", err);
    }
});
export {};
//# sourceMappingURL=background.js.map