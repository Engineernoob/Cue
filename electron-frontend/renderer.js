// --- WebSocket Setup ---
const socket = new WebSocket("ws://localhost:8000/ws");

socket.onopen = () => {
  console.log("✅ Connected to backend");
};

socket.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log("📥 From backend:", message);

  switch (message.type) {
    case "llm_response_chunk":
      responseText.textContent += message.text_chunk || "";
      break;
    case "llm_response_complete":
      hideThinking();
      break;
    case "live_summary":
      const el = document.getElementById("summary-text");
      if (el) el.textContent = message.text;
      break;
    case "ocr_result":
    case "transcript":
      inputBox.value = message.text;
      break;
    case "error":
      hideThinking();
      showResponse("Sorry, something went wrong.", "Error");
      break;
    default:
      console.warn("Unhandled message type:", message);
  }
};

// --- Element Refs ---
const inputBar = document.getElementById("input-bar");
const inputBox = document.getElementById("chat-input-box");
const sendBtn = document.getElementById("chat-send-btn");

const responseBox = document.getElementById("response-box");
const responseText = document.getElementById("response-content");
const responseStatus = document.getElementById("response-status");
const copyBtn = document.getElementById("copy-btn");
const closeBtn = document.getElementById("close-btn");

const thinkingBar = document.getElementById("thinking-bar");
const thinkingText = document.getElementById("thinking-text");
const closeThinkingBtn = document.querySelector(".close-icon");

const promptBox = document.getElementById("prompt-box");

// --- Audio Session Handling ---
let mediaRecorder = null;

// --- Ask Button: Show Input Bar ---
document.getElementById("ask-btn")?.addEventListener("click", () => {
  inputBar.hidden = !inputBar.hidden;
  if (!inputBar.hidden) inputBox.focus();
});

// --- Toggle Keyboard Shortcut (⌘ or Ctrl + \) ---
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "\\") {
    inputBar.hidden = !inputBar.hidden;
    if (!inputBar.hidden) inputBox.focus();
  }
});

// --- Chat Input Handlers ---
sendBtn?.addEventListener("click", sendChatMessage);
inputBox?.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendChatMessage();
  }
});

function sendChatMessage() {
  const text = inputBox.value.trim();
  if (!text || socket.readyState !== WebSocket.OPEN) return;

  inputBox.value = "";
  showThinking(text);
  showResponse("Thinking…");

  socket.send(
    JSON.stringify({
      type: "llm_query",
      query: text,
    })
  );
}

// --- Audio Capture Logic ---
async function startAudioCapture() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64data = reader.result.split(",")[1];
          socket.send(
            JSON.stringify({
              type: "audio_chunk",
              data: base64data,
            })
          );
        };
        reader.readAsDataURL(event.data);
      }
    };

    mediaRecorder.start(500); // Longer intervals for better performance
  } catch (err) {
    console.error("Microphone error:", err);
  }
}

function stopAudioCapture() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    mediaRecorder = null;
  }
}

// --- Thinking Display ---
function showThinking(text = "") {
  if (thinkingText) thinkingText.textContent = `“${text}”`;
  if (thinkingBar) {
    thinkingBar.hidden = false;
    thinkingBar.classList.add("show");
  }
}

function hideThinking() {
  if (thinkingBar) {
    thinkingBar.classList.remove("show");
    setTimeout(() => (thinkingBar.hidden = true), 600);
  }
}

closeThinkingBtn?.addEventListener("click", hideThinking);

// --- Response Display ---
function showResponse(content, status = "Cue's answer:") {
  if (responseText) responseText.textContent = content;
  if (responseStatus) responseStatus.textContent = status;
  if (responseBox) {
    responseBox.classList.remove("smooth-disappear");
    responseBox.classList.add("smooth-appear");
    responseBox.hidden = false;
  }
}

function hideResponse() {
  if (responseBox) {
    responseBox.classList.remove("smooth-appear");
    responseBox.classList.add("smooth-disappear");
    setTimeout(() => {
      responseBox.hidden = true;
      responseBox.classList.remove("smooth-disappear");
    }, 300);
  }
}

closeBtn?.addEventListener("click", hideResponse);

copyBtn?.addEventListener("click", () => {
  navigator.clipboard.writeText(responseText.textContent.trim());
});

// --- IPC Renderer Integration ---
const { ipcRenderer } = window.electron || {};

ipcRenderer?.on("show-input", () => {
  inputBar.hidden = false;
  inputBox.focus();
});

ipcRenderer?.on("hide-response", () => {
  hideResponse();
});

// Application state
let screenMonitoringActive = false;
let stealthModeActive = false;
let audioListeningActive = false;
let backendConnected = false;

// UI Status Management
function updateUIStatus() {
  // Audio status with Glass-inspired effects
  const audioBtn = document.getElementById("listen-btn");
  const audioStatus = document.getElementById("audio-status");
  if (audioBtn && audioStatus) {
    audioBtn.classList.toggle("active", audioListeningActive);
    audioStatus.classList.toggle("active", audioListeningActive);
    audioStatus.classList.toggle("breathing", audioListeningActive);
    audioBtn.querySelector("span").textContent = audioListeningActive ? "Listening..." : "Listen";
  }
  
  // Screen monitoring status with Glass effects
  const monitorBtn = document.getElementById("monitor-btn");
  const monitorStatus = document.getElementById("monitor-status");
  if (monitorBtn && monitorStatus) {
    monitorBtn.classList.toggle("monitoring", screenMonitoringActive);
    monitorStatus.classList.toggle("monitoring", screenMonitoringActive);
    monitorStatus.classList.toggle("breathing", screenMonitoringActive);
    monitorBtn.querySelector("span").textContent = screenMonitoringActive ? "Monitoring..." : "Monitor";
  }
  
  // Stealth mode status with enhanced Glass effects
  const stealthBtn = document.getElementById("stealth-btn");
  const stealthStatus = document.getElementById("stealth-status");
  const stealthIndicator = document.getElementById("stealth-indicator");
  const mainUI = document.getElementById("main-ui");
  
  if (stealthBtn && stealthStatus) {
    stealthBtn.classList.toggle("stealth", stealthModeActive);
    stealthStatus.classList.toggle("active", stealthModeActive);
    stealthStatus.classList.toggle("breathing", stealthModeActive);
    stealthBtn.querySelector("span").textContent = stealthModeActive ? "Stealth ON" : "Stealth";
    
    // Glass-inspired auto-hide effect
    if (mainUI) {
      mainUI.classList.toggle("auto-hide", stealthModeActive);
      if (stealthModeActive) {
        mainUI.classList.add("invisible-to-capture");
      } else {
        mainUI.classList.remove("invisible-to-capture");
      }
    }
  }
  
  if (stealthIndicator) {
    stealthIndicator.classList.toggle("hidden", !stealthModeActive);
  }
  
  if (mainUI) {
    mainUI.style.display = stealthModeActive ? "none" : "flex";
  }
}

// Button event handlers
document.getElementById("listen-btn")?.addEventListener("click", () => {
  toggleAudioListening();
});

document.getElementById("monitor-btn")?.addEventListener("click", () => {
  toggleScreenMonitoring();
});

document.getElementById("ask-btn")?.addEventListener("click", () => {
  const inputBar = document.getElementById("input-bar");
  const inputBox = document.getElementById("chat-input-box");
  inputBar.hidden = !inputBar.hidden;
  if (!inputBar.hidden) inputBox.focus();
});

document.getElementById("stealth-btn")?.addEventListener("click", () => {
  toggleStealthMode();
});

// Global keyboard shortcuts
document.addEventListener("keydown", (e) => {
  // Ctrl/Cmd + Shift + M to toggle screen monitoring
  if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === "M") {
    e.preventDefault();
    toggleScreenMonitoring();
  }
  
  // Ctrl/Cmd + Shift + S to toggle stealth mode
  if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === "S") {
    e.preventDefault();
    toggleStealthMode();
  }
  
  // Ctrl/Cmd + \ to toggle input
  if ((e.metaKey || e.ctrlKey) && e.key === "\\") {
    e.preventDefault();
    const inputBar = document.getElementById("input-bar");
    const inputBox = document.getElementById("chat-input-box");
    inputBar.hidden = !inputBar.hidden;
    if (!inputBar.hidden) inputBox.focus();
  }
  
  // Space to toggle audio (when not typing)
  if (e.code === "Space" && !e.target.matches("input, textarea")) {
    e.preventDefault();
    toggleAudioListening();
  }
});

// Toggle functions
async function toggleAudioListening() {
  audioListeningActive = !audioListeningActive;
  
  if (audioListeningActive) {
    await startAudioCapture();
    showNotification("Audio listening started", "success");
  } else {
    stopAudioCapture();
    showNotification("Audio listening stopped", "info");
  }
  
  updateUIStatus();
}

async function toggleScreenMonitoring() {
  if (screenMonitoringActive) {
    await window.electronAPI?.screenMonitor?.stop();
    screenMonitoringActive = false;
    showNotification("Screen monitoring stopped", "info");
  } else {
    await window.electronAPI?.screenMonitor?.start();
    screenMonitoringActive = true;
    showNotification("AI is now watching for coding problems", "success");
  }
  
  updateUIStatus();
}

async function toggleStealthMode() {
  try {
    const newStealthState = await window.electronAPI?.stealth?.toggle();
    stealthModeActive = newStealthState;
    
    if (stealthModeActive) {
      // Enable stealth mode
      showStealthNotification("Stealth Mode Activated - Cue is now invisible");
      setTimeout(() => {
        updateUIStatus();
        // Start screen monitoring automatically in stealth mode
        if (!screenMonitoringActive) {
          screenMonitoringActive = true;
          window.electronAPI?.screenMonitor?.start();
        }
      }, 2000);
    } else {
      updateUIStatus();
      showNotification("Stealth mode disabled - Cue is now visible", "info");
    }
  } catch (error) {
    console.error('Failed to toggle stealth mode:', error);
    showNotification("Failed to toggle stealth mode", "error");
  }
}

function showStealthNotification(message) {
  const notification = document.createElement("div");
  notification.className = "stealth-notification";
  notification.textContent = message;
  document.body.appendChild(notification);
  
  setTimeout(() => {
    if (notification.parentNode) {
      document.body.removeChild(notification);
    }
  }, 3000);
}

// Listen for coding guidance from the AI
ipcRenderer?.on("coding-guidance", (event, guidance) => {
  showCodingGuidance(guidance);
});

ipcRenderer?.on("screen-monitoring-status", (event, status) => {
  screenMonitoringActive = status.monitoring;
  showNotification(status.message, status.monitoring ? "success" : "info");
});

// Send screen data for analysis
ipcRenderer?.on("send-screen-for-analysis", (event, data) => {
  if (data && data.imageData) {
    ipcRenderer?.send("image-data-chunk", data.imageData);
  }
});

function showCodingGuidance(guidance) {
  // Create or update guidance panel
  let guidancePanel = document.getElementById("coding-guidance-panel");
  
  if (!guidancePanel) {
    guidancePanel = createGuidancePanel();
    document.body.appendChild(guidancePanel);
  }
  
  updateGuidanceContent(guidancePanel, guidance);
  showGuidancePanel(guidancePanel);
}

function createGuidancePanel() {
  const panel = document.createElement("div");
  panel.id = "coding-guidance-panel";
  panel.style.cssText = `
    position: fixed;
    top: 100px;
    right: 20px;
    width: 400px;
    max-height: 500px;
    background: rgba(30, 30, 30, 0.95);
    border: 1px solid #444;
    border-radius: 8px;
    padding: 20px;
    color: #fff;
    font-family: system-ui, sans-serif;
    z-index: 10000;
    overflow-y: auto;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    backdrop-filter: blur(10px);
    display: none;
  `;
  
  // Add close button
  const closeBtn = document.createElement("button");
  closeBtn.innerHTML = "✕";
  closeBtn.style.cssText = `
    position: absolute;
    top: 10px;
    right: 10px;
    background: none;
    border: none;
    color: #888;
    font-size: 18px;
    cursor: pointer;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
  `;
  closeBtn.onmouseover = () => closeBtn.style.background = "#444";
  closeBtn.onmouseout = () => closeBtn.style.background = "none";
  closeBtn.onclick = () => hideGuidancePanel(panel);
  
  panel.appendChild(closeBtn);
  return panel;
}

function updateGuidanceContent(panel, guidance) {
  // Clear previous content (except close button)
  const closeBtn = panel.querySelector("button");
  panel.innerHTML = "";
  panel.appendChild(closeBtn);
  
  // Add guidance content
  const content = document.createElement("div");
  content.style.marginTop = "30px";
  
  if (guidance.type === "problem_guidance") {
    content.innerHTML = `
      <div style="margin-bottom: 15px;">
        <h3 style="color: #4CAF50; margin: 0 0 10px 0; font-size: 16px;">
          🎯 ${guidance.platform.toUpperCase()} Problem Detected
        </h3>
        <p style="margin: 0; line-height: 1.4; color: #ddd;">${guidance.message}</p>
      </div>
      
      <div style="margin-bottom: 15px;">
        <h4 style="color: #2196F3; margin: 0 0 8px 0; font-size: 14px;">Next Steps:</h4>
        <ul style="margin: 0; padding-left: 20px; line-height: 1.6;">
          ${guidance.next_steps.map(step => `<li style="margin-bottom: 4px;">${step}</li>`).join('')}
        </ul>
      </div>
      
      ${guidance.analysis.patterns.length > 0 ? `
        <div style="background: rgba(76, 175, 80, 0.1); padding: 10px; border-radius: 4px; border-left: 3px solid #4CAF50;">
          <strong style="color: #4CAF50;">Patterns Detected:</strong> ${guidance.analysis.patterns.join(', ').replace(/_/g, ' ')}
        </div>
      ` : ''}
    `;
  } else if (guidance.type === "progressive_hint") {
    content.innerHTML = `
      <div style="margin-bottom: 15px;">
        <h3 style="color: #FF9800; margin: 0 0 10px 0; font-size: 16px;">
          💡 Hint #${guidance.hint_level}
        </h3>
        <p style="margin: 0; line-height: 1.4; color: #ddd;">${guidance.message}</p>
      </div>
      
      ${guidance.code_template ? `
        <div style="margin-bottom: 15px;">
          <h4 style="color: #9C27B0; margin: 0 0 8px 0; font-size: 14px;">Code Template:</h4>
          <pre style="background: #1a1a1a; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 12px; margin: 0;"><code>${guidance.code_template}</code></pre>
        </div>
      ` : ''}
      
      <div style="background: rgba(33, 150, 243, 0.1); padding: 10px; border-radius: 4px; border-left: 3px solid #2196F3;">
        <strong style="color: #2196F3;">Remember:</strong> ${guidance.encouragement}
      </div>
    `;
  } else if (guidance.type === "solution_walkthrough") {
    const walkthrough = guidance.walkthrough;
    content.innerHTML = `
      <div style="margin-bottom: 15px;">
        <h3 style="color: #f44336; margin: 0 0 10px 0; font-size: 16px;">
          🔍 Solution Walkthrough
        </h3>
        <div style="background: rgba(244, 67, 54, 0.1); padding: 10px; border-radius: 4px; margin-bottom: 15px;">
          <strong style="color: #f44336;">⚠️ ${guidance.warning}</strong>
        </div>
      </div>
      
      <div style="margin-bottom: 15px;">
        <h4 style="color: #4CAF50; margin: 0 0 8px 0; font-size: 14px;">Approach: ${walkthrough.approach}</h4>
        <ol style="margin: 0; padding-left: 20px; line-height: 1.6;">
          ${walkthrough.steps.map(step => `<li style="margin-bottom: 4px;">${step}</li>`).join('')}
        </ol>
      </div>
      
      <div style="background: rgba(156, 39, 176, 0.1); padding: 10px; border-radius: 4px; margin-bottom: 10px;">
        <strong style="color: #9C27B0;">Complexity:</strong> ${walkthrough.complexity}
      </div>
      
      <div style="background: rgba(33, 150, 243, 0.1); padding: 10px; border-radius: 4px;">
        <strong style="color: #2196F3;">Pro Tip:</strong> ${walkthrough.next_step}
      </div>
    `;
  }
  
  panel.appendChild(content);
}

function showGuidancePanel(panel) {
  panel.style.display = "block";
  panel.style.opacity = "0";
  panel.style.transform = "translateX(20px)";
  
  requestAnimationFrame(() => {
    panel.style.transition = "all 0.3s ease-out";
    panel.style.opacity = "1";
    panel.style.transform = "translateX(0)";
  });
}

function hideGuidancePanel(panel) {
  panel.style.transition = "all 0.3s ease-out";
  panel.style.opacity = "0";
  panel.style.transform = "translateX(20px)";
  
  setTimeout(() => {
    panel.style.display = "none";
  }, 300);
}

function showNotification(message, type = "info") {
  const notification = document.createElement("div");
  const colors = {
    success: "#4CAF50",
    error: "#f44336", 
    info: "#2196F3",
    warning: "#FF9800"
  };
  
  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: ${colors[type] || colors.info};
    color: white;
    padding: 12px 20px;
    border-radius: 4px;
    font-size: 14px;
    z-index: 10001;
    opacity: 0;
    transform: translateX(100px);
    transition: all 0.3s ease-out;
  `;
  
  notification.textContent = message;
  document.body.appendChild(notification);
  
  requestAnimationFrame(() => {
    notification.style.opacity = "1";
    notification.style.transform = "translateX(0)";
  });
  
  setTimeout(() => {
    notification.style.opacity = "0";
    notification.style.transform = "translateX(100px)";
    setTimeout(() => document.body.removeChild(notification), 300);
  }, 4000);
}

// Initialize UI on load
document.addEventListener('DOMContentLoaded', () => {
  updateUIStatus();
});

// Also initialize when window loads
window.addEventListener('load', () => {
  updateUIStatus();
  checkBackendStatus();
});

// Backend status monitoring
async function checkBackendStatus() {
  try {
    const status = await window.electronAPI?.backend?.getStatus();
    backendConnected = status?.isRunning || false;
    updateUIStatus();
    
    if (!backendConnected && status?.pythonFound === false) {
      showNotification("Python not found. Please install Python 3.8+", "error");
    } else if (!backendConnected) {
      showNotification("Backend starting...", "info");
    }
  } catch (error) {
    console.error('Failed to check backend status:', error);
  }
}

// Listen for backend connection status
socket.onopen = () => {
  console.log("✅ Connected to backend");
  backendConnected = true;
  updateUIStatus();
  showNotification("AI backend connected", "success");
};

socket.onclose = () => {
  console.log("🔌 Disconnected from backend");  
  backendConnected = false;
  updateUIStatus();
  showNotification("Backend disconnected - retrying...", "warning");
};

socket.onerror = () => {
  backendConnected = false;
  updateUIStatus();
};
