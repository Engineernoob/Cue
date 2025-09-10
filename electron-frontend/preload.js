// preload.js
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  // Backend
  onBackendStatus: (callback) =>
    ipcRenderer.on("backend-status", (_e, status) => callback(status)),
  onBackendMessage: (callback) =>
    ipcRenderer.on("backend-message", (_e, message) => callback(message)),

  // Audio / Screen
  onToggleAudioCapture: (callback) =>
    ipcRenderer.on("toggle-audio-capture", () => callback()),
  onToggleScreenCapture: (callback) =>
    ipcRenderer.on("toggle-screen-capture", () => callback()),
  onAudioStatus: (callback) =>
    ipcRenderer.on("audio-status", (_e, status) => callback(status)),
  onScreenStatus: (callback) =>
    ipcRenderer.on("screen-status", (_e, status) => callback(status)),

  // LLM
  sendLlmQuery: (query) => ipcRenderer.send("send-llm-query", query),
  onLlmResponseError: (callback) =>
    ipcRenderer.on("llm-response-error", (_e, err) => callback(err)),

  // Capture
  sendAudioChunkData: (data) => ipcRenderer.send("audio-chunk-data", data),
  sendImageChunkData: (data) => ipcRenderer.send("image-data-chunk", data),
  getDesktopSources: (options) =>
    ipcRenderer.invoke("get-desktop-sources", options),

  // Window
  onWindowVisibility: (callback) =>
    ipcRenderer.on("window-visibility", (_e, status) => callback(status)),
  requestHideWindow: () => ipcRenderer.send("request-hide-window"),

  // Push-To-Talk (renderer handles Shift+Space)
  onPushToTalkStart: (callback) =>
    ipcRenderer.on("ptt-start", () => callback()),
  onPushToTalkStop: (callback) => ipcRenderer.on("ptt-stop", () => callback()),

  // Auto-Coach + Answer Style
  onAutoCoachToggle: (callback) =>
    ipcRenderer.on("auto-coach-toggle", () => callback()),
  onAnswerStyleCycle: (callback) =>
    ipcRenderer.on("answer-style-cycle", () => callback()),

  // Screen monitor
  screenMonitor: {
    start: () => ipcRenderer.invoke("screen-monitor:start"),
    stop: () => ipcRenderer.invoke("screen-monitor:stop"),
    getSolutionWalkthrough: () =>
      ipcRenderer.invoke("screen-monitor:get-solution-walkthrough"),
    updateProgress: (stage) =>
      ipcRenderer.invoke("screen-monitor:update-progress", stage),
  },
  onCodingGuidance: (callback) =>
    ipcRenderer.on("coding-guidance", (_e, g) => callback(g)),
  onScreenMonitoringStatus: (callback) =>
    ipcRenderer.on("screen-monitoring-status", (_e, s) => callback(s)),

  // Stealth
  stealth: {
    enable: () => ipcRenderer.invoke("stealth:enable"),
    disable: () => ipcRenderer.invoke("stealth:disable"),
    toggle: () => ipcRenderer.invoke("stealth:toggle"),
  },

  // Backend manager
  backend: {
    getStatus: () => ipcRenderer.invoke("backend:status"),
    restart: () => ipcRenderer.invoke("backend:restart"),
    healthCheck: () => ipcRenderer.invoke("backend:health"),
    stop: () => ipcRenderer.invoke("backend:stop"),
  },
});
