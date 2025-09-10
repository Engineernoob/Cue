const {
  app,
  BrowserWindow,
  ipcMain,
  desktopCapturer,
  screen,
} = require("electron");
const path = require("path");
const WebSocket = require("ws");

const { registerSessionHandlers } = require("./sessionManager");
const { registerCoachingHandlers } = require("./coachingEngine");
const { registerNeurodivergentHandlers } = require("./neurodivergentSupport");
const { registerCodingAssessmentHandlers } = require("./codingAssessmentHelper");
const { createScreenMonitor } = require("./screenMonitor");
const { createStealthManager, registerStealthHandlers } = require("./stealthMode");
const backendManager = require("./backendManager");
const { getConfig, setConfig } = require("./config");

const PYTHON_BACKEND_WS_URL = getConfig('backend.wsUrl');

let mainWindow;
let ws;
let audioCapturing = false;

// --- Create Minimal Floating Bar UI ---
function createWindow() {
  const { width: screenWidth, height: screenHeight } = screen.getPrimaryDisplay().workAreaSize;
  const uiConfig = getConfig('ui');
  const windowWidth = 500;
  const windowHeight = 90;

  // Resolve position keywords (center, bottom[-offset])
  const posXRaw = uiConfig.position?.x ?? 'center';
  const posYRaw = uiConfig.position?.y ?? 'bottom-40';

  const resolveX = () => {
    if (posXRaw === 'center') return Math.round((screenWidth - windowWidth) / 2);
    if (typeof posXRaw === 'number') return posXRaw;
    return 0;
  };

  const resolveY = () => {
    if (typeof posYRaw === 'number') return posYRaw;
    if (typeof posYRaw === 'string') {
      if (posYRaw === 'center') return Math.round((screenHeight - windowHeight) / 2);
      if (posYRaw.startsWith('bottom')) {
        const parts = posYRaw.split('-');
        const offset = parts.length > 1 ? Math.max(0, parseInt(parts[1], 10) || 0) : 40;
        return Math.max(0, screenHeight - windowHeight - offset);
      }
    }
    return 30; // fallback near top
  };

  mainWindow = new BrowserWindow({
    width: windowWidth,
    height: windowHeight,
    x: resolveX(),
    y: resolveY(),
    frame: false,
    transparent: true,
    alwaysOnTop: uiConfig.alwaysOnTop,
    skipTaskbar: true,
    resizable: true,
    movable: true,
    hasShadow: false,
    show: true,
    opacity: uiConfig.opacity,
    vibrancy: process.platform === 'darwin' ? 'ultra-dark' : undefined,
    backgroundMaterial: process.platform === 'win32' ? 'acrylic' : undefined,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true,
      backgroundThrottling: false,
      experimentalFeatures: true,
    },
  });

  // Persist window position after user moves it
  let savePosTimer = null;
  const scheduleSavePos = () => {
    if (savePosTimer) clearTimeout(savePosTimer);
    savePosTimer = setTimeout(() => {
      try {
        if (!mainWindow || mainWindow.isDestroyed() || !mainWindow.isVisible()) return;
        const { x, y } = mainWindow.getBounds();
        // Ignore stealth off-screen coords
        if (x < -5000 || y < -5000) return;
        const current = getConfig('ui.position');
        if (!current || current.x !== x || current.y !== y) {
          setConfig('ui.position', { x, y });
        }
      } catch (e) {
        console.error('Failed to save window position:', e);
      }
    }, 300);
  };

  mainWindow.on('move', scheduleSavePos);
  mainWindow.on('moved', scheduleSavePos);

  if (process.platform === "darwin") {
    mainWindow.setAlwaysOnTop(true, "floating");
    app.dock.hide();
    mainWindow.setContentProtection(true);
  }

  mainWindow.loadFile("index.html");

  mainWindow.on("closed", () => {
    mainWindow = null;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.close();
    }
  });
}

// --- Lifecycle ---
let neurodivergentSupport;
let codingAssessmentHelper;
let screenMonitor;
let stealthManager;

app.whenReady().then(async () => {
  createWindow();
  
  // Start backend first, then connect
  try {
    console.log('🚀 Starting integrated backend...');
    await backendManager.startBackend();
    
    // Wait a moment for backend to fully initialize
    setTimeout(() => {
      connectToPythonBackend();
    }, 2000);
    
  } catch (error) {
    console.error('❌ Failed to start backend:', error);
    // Continue without backend - user will see connection error
    connectToPythonBackend();
  }
  
  // Initialize all managers
  registerSessionHandlers();
  registerCoachingHandlers();
  neurodivergentSupport = registerNeurodivergentHandlers(mainWindow);
  codingAssessmentHelper = registerCodingAssessmentHandlers(mainWindow);
  screenMonitor = createScreenMonitor(mainWindow);
  stealthManager = createStealthManager(mainWindow);
  
  // Register IPC handlers
  registerScreenMonitorHandlers();
  registerBackendHandlers();
  const stealthHotkeys = registerStealthHandlers(stealthManager);
  registerGlobalShortcuts(stealthHotkeys);
  registerPushToTalkShortcuts();
  registerAutoCoachShortcut();
  registerAnswerStyleShortcut();
  
  // Start health monitoring
  backendManager.startHealthMonitoring();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
    if (mainWindow && !mainWindow.isVisible()) mainWindow.hide();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("will-quit", () => {
  // No global shortcuts to unregister now
  if (process.platform === "darwin" && app.dock.isVisible()) {
    app.dock.show();
  }
});

// --- WebSocket Sync ---
function connectToPythonBackend() {
  ws = new WebSocket(PYTHON_BACKEND_WS_URL);

  ws.onopen = () => {
    console.log("✅ Connected to Python backend");
    mainWindow?.webContents.send("backend-status", { connected: true });
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    mainWindow?.webContents.send("backend-message", msg);

    if (msg.type === "coaching_prompt") {
      mainWindow?.webContents.send("show-coaching-prompt", msg.payload);
    }

    // Handle coding problem detection from backend
    if (msg.type === "coding_problem_detected") {
      screenMonitor?.onProblemDetected(msg);
    }

    if (!mainWindow?.isVisible()) {
      mainWindow?.show();
      mainWindow?.webContents.send("window-visibility", { visible: true });
    }
  };

  ws.onclose = (e) => {
    console.warn("🔌 Disconnected from backend:", e.code, e.reason);
    mainWindow?.webContents.send("backend-status", { connected: false });
    mainWindow?.hide();
    
    // Exponential backoff for reconnection
    const backoffDelay = Math.min(getConfig('backend.reconnectDelay') * Math.pow(2, (e.attempts || 0)), 30000);
    setTimeout(() => connectToPythonBackend(e.attempts ? e.attempts + 1 : 1), backoffDelay);
  };

  ws.onerror = (err) => {
    console.error("WebSocket Error:", err.message);
    mainWindow?.webContents.send("backend-status", {
      connected: false,
      error: err.message,
    });
    ws.close();
  };
}

// --- IPC Forwarding ---
ipcMain.on("toggle-audio-capture", (_event, enabled) => {
  if (typeof enabled === "boolean") {
    audioCapturing = enabled;
  } else {
    // fallback: toggle if no param
    audioCapturing = !audioCapturing;
  }

  mainWindow?.webContents.send("audio-status", { capturing: audioCapturing });

  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(
      JSON.stringify({ type: "audio_capture_toggle", enabled: audioCapturing })
    );
  }
});

ipcMain.on("audio-chunk-data", (event, data) => {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "audio_chunk", data }));
  } else {
    event.sender.send("audio-status", {
      capturing: false,
      error: "Backend not connected",
    });
  }
});

ipcMain.on("image-data-chunk", (event, data) => {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "image_data", data }));
  } else {
    event.sender.send("screen-status", {
      capturing: false,
      error: "Backend not connected",
    });
  }
});

ipcMain.on("send-llm-query", (event, query) => {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "llm_query", query, trigger: "manual" }));
  } else {
    mainWindow?.webContents.send("llm-response-error", {
      message: "Backend not connected. Cannot send query.",
    });
  }
});

ipcMain.on("request-hide-window", () => {
  if (mainWindow?.isVisible()) {
    mainWindow.hide();
    mainWindow.webContents.send("window-visibility", { visible: false });
  }
});

// --- Desktop Sources Handler ---
ipcMain.handle("get-desktop-sources", async (_event, options) => {
  const sources = await desktopCapturer.getSources(options);
  return sources.map((src) => ({
    id: src.id,
    name: src.name,
    display_id: src.display_id,
    appIcon: src.appIcon?.toDataURL() ?? null,
    thumbnail: src.thumbnail?.toDataURL() ?? null,
  }));
});

// --- Screen Monitor IPC Handlers ---
function registerScreenMonitorHandlers() {
  ipcMain.handle('screen-monitor:start', () => {
    screenMonitor?.startMonitoring();
    return true;
  });

  ipcMain.handle('screen-monitor:stop', () => {
    screenMonitor?.stopMonitoring();
    return true;
  });

  ipcMain.handle('screen-monitor:get-solution-walkthrough', () => {
    screenMonitor?.provideSolutionWalkthrough();
    return true;
  });

  ipcMain.handle('screen-monitor:update-progress', (_event, stage) => {
    screenMonitor?.updateProgress(stage);
    return true;
  });

  // Handle screen analysis requests from renderer
  ipcMain.on('send-screen-for-analysis', (_event, data) => {
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ 
        type: "image_data", 
        data: data.imageData,
        context: 'screen_monitoring'
      }));
    }
  });
}

// --- Global Shortcuts ---
function registerGlobalShortcuts(stealthHotkeys) {
  const { globalShortcut } = require('electron');
  
  try {
    // Screen monitoring toggle
    globalShortcut.register('CommandOrControl+Shift+M', () => {
      console.log('Global shortcut: Screen monitoring toggle');
      if (screenMonitor?.isMonitoring) {
        screenMonitor.stopMonitoring();
      } else {
        screenMonitor?.startMonitoring();
      }
    });

    // Stealth mode toggle
    globalShortcut.register(stealthHotkeys['stealth-toggle'], () => {
      console.log('Global shortcut: Stealth mode toggle');
      stealthManager?.handleStealthHotkey('stealth-toggle');
    });

    // Stealth guidance
    globalShortcut.register(stealthHotkeys['stealth-guidance'], () => {
      console.log('Global shortcut: Stealth guidance');
      stealthManager?.handleStealthHotkey('stealth-guidance');
    });

    // Stealth hint
    globalShortcut.register(stealthHotkeys['stealth-hint'], () => {
      console.log('Global shortcut: Stealth hint');
      stealthManager?.handleStealthHotkey('stealth-hint');
    });

    // General input toggle
    globalShortcut.register('CommandOrControl+\\', () => {
      console.log('Global shortcut: Input toggle');
      mainWindow?.webContents.send('toggle-input');
    });

  } catch (error) {
    console.error('Failed to register global shortcuts:', error);
  }
}

// --- Push-To-Talk Global Shortcuts ---
function registerPushToTalkShortcuts() {
  const { globalShortcut } = require('electron');
  const pttConfig = getConfig('hotkeys.pushToTalk');

  try {
    // Start recording while holding a dedicated key combo
    globalShortcut.register(pttConfig.start, () => {
      console.log('Global shortcut: PTT start');
      mainWindow?.webContents.send('ptt-start');
    });

    // Stop recording when pressing stop combo (limitation: keyup not available globally)
    globalShortcut.register(pttConfig.stop, () => {
      console.log('Global shortcut: PTT stop');
      mainWindow?.webContents.send('ptt-stop');
    });

  } catch (error) {
    console.error('Failed to register push-to-talk shortcuts:', error);
  }
}

// --- Auto-Coach Toggle (global) ---
function registerAutoCoachShortcut() {
  const { globalShortcut } = require('electron');
  const hotkey = getConfig('hotkeys.autoCoachToggle');
  try {
    globalShortcut.register(hotkey, () => {
      console.log('Global shortcut: Auto-Coach toggle');
      mainWindow?.webContents.send('auto-coach-toggle');
    });
  } catch (error) {
    console.error('Failed to register auto-coach shortcut:', error);
  }
}

// --- Answer Style Cycle (global) ---
function registerAnswerStyleShortcut() {
  const { globalShortcut } = require('electron');
  const hotkey = getConfig('hotkeys.answerStyleCycle');
  try {
    globalShortcut.register(hotkey, () => {
      console.log('Global shortcut: Answer Style cycle');
      mainWindow?.webContents.send('answer-style-cycle');
    });
  } catch (error) {
    console.error('Failed to register answer style shortcut:', error);
  }
}

// --- Backend IPC Handlers ---
function registerBackendHandlers() {
  ipcMain.handle('backend:status', () => {
    return backendManager.getStatus();
  });

  ipcMain.handle('backend:restart', async () => {
    try {
      await backendManager.restartBackend();
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  });

  ipcMain.handle('backend:health', async () => {
    return await backendManager.healthCheck();
  });

  ipcMain.handle('backend:stop', () => {
    backendManager.stopBackend();
    return { success: true };
  });
}
