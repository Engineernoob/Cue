// main.js
const {
  app,
  BrowserWindow,
  ipcMain,
  desktopCapturer,
  screen,
  globalShortcut,
} = require("electron");
const path = require("path");
const WebSocket = require("ws");

const { registerSessionHandlers } = require("./sessionManager");
const { registerCoachingHandlers } = require("./coachingEngine");
const { registerNeurodivergentHandlers } = require("./neurodivergentSupport");
const {
  registerCodingAssessmentHandlers,
} = require("./codingAssessmentHelper");
const { createScreenMonitor } = require("./screenMonitor");
const {
  createStealthManager,
  registerStealthHandlers,
} = require("./stealthMode");
const backendManager = require("./backendManager");
const { getConfig, setConfig } = require("./config");

const PYTHON_BACKEND_WS_URL = getConfig("backend.wsUrl");

let mainWindow;
let ws;
let audioCapturing = false;
let reconnectAttempts = 0;

// --- Create Minimal Floating Bar UI ---
function createWindow() {
  const { width: screenWidth, height: screenHeight } =
    screen.getPrimaryDisplay().workAreaSize;
  const uiConfig = getConfig("ui");
  const windowWidth = 500;
  const windowHeight = 90;

  const posXRaw = uiConfig.position?.x ?? "center";
  const posYRaw = uiConfig.position?.y ?? "bottom-40";

  const resolveX = () => {
    if (posXRaw === "center")
      return Math.round((screenWidth - windowWidth) / 2);
    if (typeof posXRaw === "number") return posXRaw;
    return 0;
  };

  const resolveY = () => {
    if (typeof posYRaw === "number") return posYRaw;
    if (typeof posYRaw === "string") {
      if (posYRaw === "center")
        return Math.round((screenHeight - windowHeight) / 2);
      if (posYRaw.startsWith("bottom")) {
        const parts = posYRaw.split("-");
        const offset =
          parts.length > 1 ? Math.max(0, parseInt(parts[1], 10) || 0) : 40;
        return Math.max(0, screenHeight - windowHeight - offset);
      }
    }
    return 30;
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
    vibrancy: process.platform === "darwin" ? "ultra-dark" : undefined,
    backgroundMaterial: process.platform === "win32" ? "acrylic" : undefined,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true,
      backgroundThrottling: false,
      experimentalFeatures: true,
    },
  });

  let savePosTimer = null;
  const scheduleSavePos = () => {
    if (savePosTimer) clearTimeout(savePosTimer);
    savePosTimer = setTimeout(() => {
      try {
        if (!mainWindow || mainWindow.isDestroyed() || !mainWindow.isVisible())
          return;
        const { x, y } = mainWindow.getBounds();
        if (x < -5000 || y < -5000) return;
        const current = getConfig("ui.position");
        if (!current || current.x !== x || current.y !== y) {
          setConfig("ui.position", { x, y });
        }
      } catch (e) {
        console.error("Failed to save window position:", e);
      }
    }, 300);
  };

  mainWindow.on("move", scheduleSavePos);
  mainWindow.on("moved", scheduleSavePos);

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

  try {
    console.log("🚀 Starting integrated backend...");
    await backendManager.startBackend();
    setTimeout(connectToPythonBackend, 2000);
  } catch (error) {
    console.error("❌ Failed to start backend:", error);
    connectToPythonBackend();
  }

  registerSessionHandlers();
  registerCoachingHandlers();
  neurodivergentSupport = registerNeurodivergentHandlers(mainWindow);
  codingAssessmentHelper = registerCodingAssessmentHandlers(mainWindow);
  screenMonitor = createScreenMonitor(mainWindow);
  stealthManager = createStealthManager(mainWindow);

  registerScreenMonitorHandlers();
  registerBackendHandlers();
  const stealthHotkeys = registerStealthHandlers(stealthManager);
  registerGlobalShortcuts(stealthHotkeys);
  registerAutoCoachShortcut();
  registerAnswerStyleShortcut();

  backendManager.startHealthMonitoring();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
    if (mainWindow && !mainWindow.isVisible()) mainWindow.show();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

// --- WebSocket Sync ---
function connectToPythonBackend() {
  ws = new WebSocket(PYTHON_BACKEND_WS_URL);

  ws.onopen = () => {
    reconnectAttempts = 0;
    console.log("✅ Connected to Python backend");
    mainWindow?.webContents.send("backend-status", { connected: true });
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    mainWindow?.webContents.send("backend-message", msg);

    if (msg.type === "coaching_prompt") {
      mainWindow?.webContents.send("show-coaching-prompt", msg.payload);
    }

    if (msg.type === "coding_problem_detected") {
      screenMonitor?.onProblemDetected(msg);
    }
  };

  ws.onclose = () => {
    reconnectAttempts++;
    mainWindow?.webContents.send("backend-status", { connected: false });
    const baseDelay = getConfig("backend.reconnectDelay") || 1000;
    const backoffDelay = Math.min(
      baseDelay * Math.pow(2, reconnectAttempts),
      30000
    );
    console.warn(`🔌 Disconnected. Reconnecting in ${backoffDelay}ms`);
    setTimeout(connectToPythonBackend, backoffDelay);
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
  audioCapturing = typeof enabled === "boolean" ? enabled : !audioCapturing;
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
    event.sender.send("llm-response-error", {
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
  ipcMain.handle("screen-monitor:start", () => {
    screenMonitor?.startMonitoring();
    return true;
  });

  ipcMain.handle("screen-monitor:stop", () => {
    screenMonitor?.stopMonitoring();
    return true;
  });

  ipcMain.handle("screen-monitor:get-solution-walkthrough", () => {
    screenMonitor?.provideSolutionWalkthrough();
    return true;
  });

  ipcMain.handle("screen-monitor:update-progress", (_event, stage) => {
    screenMonitor?.updateProgress(stage);
    return true;
  });

  ipcMain.on("send-screen-for-analysis", (_event, data) => {
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "image_data",
          data: data.imageData,
          context: "screen_monitoring",
        })
      );
    }
  });
}

// --- Global Shortcuts ---
function registerGlobalShortcuts(stealthHotkeys) {
  try {
    globalShortcut.register("CommandOrControl+Shift+M", () => {
      if (screenMonitor?.isMonitoring) {
        screenMonitor.stopMonitoring();
      } else {
        screenMonitor?.startMonitoring();
      }
    });

    globalShortcut.register(stealthHotkeys["stealth-toggle"], () => {
      stealthManager?.handleStealthHotkey("stealth-toggle");
    });

    globalShortcut.register(stealthHotkeys["stealth-guidance"], () => {
      stealthManager?.handleStealthHotkey("stealth-guidance");
    });

    globalShortcut.register(stealthHotkeys["stealth-hint"], () => {
      stealthManager?.handleStealthHotkey("stealth-hint");
    });

    globalShortcut.register("CommandOrControl+\\", () => {
      mainWindow?.webContents.send("toggle-input");
    });
  } catch (error) {
    console.error("Failed to register global shortcuts:", error);
  }
}

// --- Auto-Coach Toggle ---
function registerAutoCoachShortcut() {
  const hotkey = getConfig("hotkeys.autoCoachToggle");
  try {
    globalShortcut.register(hotkey, () => {
      mainWindow?.webContents.send("auto-coach-toggle");
    });
  } catch (error) {
    console.error("Failed to register auto-coach shortcut:", error);
  }
}

// --- Answer Style Cycle ---
function registerAnswerStyleShortcut() {
  const hotkey = getConfig("hotkeys.answerStyleCycle");
  try {
    globalShortcut.register(hotkey, () => {
      mainWindow?.webContents.send("answer-style-cycle");
    });
  } catch (error) {
    console.error("Failed to register answer style shortcut:", error);
  }
}

// --- Backend IPC Handlers ---
function registerBackendHandlers() {
  ipcMain.handle("backend:status", () => backendManager.getStatus());

  ipcMain.handle("backend:restart", async () => {
    try {
      await backendManager.restartBackend();
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  });

  ipcMain.handle("backend:health", async () => {
    return await backendManager.healthCheck();
  });

  ipcMain.handle("backend:stop", () => {
    backendManager.stopBackend();
    return { success: true };
  });
}
