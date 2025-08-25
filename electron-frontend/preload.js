// electron-frontend/preload.js

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    // Backend status and messages
    onBackendStatus: (callback) => ipcRenderer.on('backend-status', (event, status) => callback(status)),
    onBackendMessage: (callback) => ipcRenderer.on('backend-message', (event, message) => callback(message)),

    // Audio/Screen capture controls (triggered by hotkeys from main)
    onToggleAudioCapture: (callback) => ipcRenderer.on('toggle-audio-capture', (event) => callback()),
    onToggleScreenCapture: (callback) => ipcRenderer.on('toggle-screen-capture', (event) => callback()),

    // Status updates from main process (e.g., if capture fails)
    onAudioStatus: (callback) => ipcRenderer.on('audio-status', (event, status) => callback(status)),
    onScreenStatus: (callback) => ipcRenderer.on('screen-status', (event, status) => callback(status)),

    // LLM interaction
    sendLlmQuery: (query) => ipcRenderer.send('send-llm-query', query),
    onTriggerLlmInput: (callback) => ipcRenderer.on('trigger-llm-input', (event) => callback()),
    onLlmResponseError: (callback) => ipcRenderer.on('llm-response-error', (event, error) => callback(error)),

    // Methods to send captured data from renderer to main
    sendAudioChunkData: (data) => ipcRenderer.send('audio-chunk-data', data),
    sendImageChunkData: (data) => ipcRenderer.send('image-data-chunk', data),

    // Expose desktopCapturer.getSources (via IPC handle)
    getDesktopSources: (options) => ipcRenderer.invoke('get-desktop-sources', options),

    // Window visibility controls
    onWindowVisibility: (callback) => ipcRenderer.on('window-visibility', (event, status) => callback(status)),
    requestHideWindow: () => ipcRenderer.send('request-hide-window'),

    // Screen monitoring API
    screenMonitor: {
        start: () => ipcRenderer.invoke('screen-monitor:start'),
        stop: () => ipcRenderer.invoke('screen-monitor:stop'),
        getSolutionWalkthrough: () => ipcRenderer.invoke('screen-monitor:get-solution-walkthrough'),
        updateProgress: (stage) => ipcRenderer.invoke('screen-monitor:update-progress', stage)
    },

    // Coding guidance callbacks
    onCodingGuidance: (callback) => ipcRenderer.on('coding-guidance', (event, guidance) => callback(guidance)),
    onScreenMonitoringStatus: (callback) => ipcRenderer.on('screen-monitoring-status', (event, status) => callback(status))
});