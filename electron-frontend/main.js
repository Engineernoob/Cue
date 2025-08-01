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

const PYTHON_BACKEND_WS_URL = "ws://127.0.0.1:8000/ws";

let mainWindow;
let ws;
let audioCapturing = false;

// --- Create Minimal Floating Bar UI ---
function createWindow() {
  const { width: screenWidth } = screen.getPrimaryDisplay().workAreaSize;
  const windowWidth = 500;
  const windowHeight = 90;

  mainWindow = new BrowserWindow({
    width: windowWidth,
    height: windowHeight,
    x: Math.round((screenWidth - windowWidth) / 2),
    y: 30,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    skipTaskbar: true,
    resizable: true,
    movable: true,
    hasShadow: false,
    show: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true,
      backgroundThrottling: false,
    },
  });

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
app.whenReady().then(() => {
  createWindow();
  connectToPythonBackend();
  // Removed registerGlobalShortcuts();
  registerSessionHandlers();
  registerCoachingHandlers();

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

    if (!mainWindow?.isVisible()) {
      mainWindow?.show();
      mainWindow?.webContents.send("window-visibility", { visible: true });
    }
  };

  ws.onclose = (e) => {
    console.warn("🔌 Disconnected from backend:", e.code, e.reason);
    mainWindow?.webContents.send("backend-status", { connected: false });
    mainWindow?.hide();
    setTimeout(connectToPythonBackend, 5000);
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
