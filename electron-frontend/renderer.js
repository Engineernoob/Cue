// --- WebSocket Setup ---
let socket = null;
let reconnectAttempts = 0;
let backendConnected = false;
let answerStyle = "PAR";

function setupSocket() {
  socket = new WebSocket("ws://localhost:8001/ws");

  socket.onopen = () => {
    console.log("✅ Connected to backend");
    backendConnected = true;
    updateUIStatus();
    showNotification("AI backend connected", "success");
    reconnectAttempts = 0;
    appendMessage("system", "✅ Connected to backend");
  };

  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    console.log("📥 From backend:", message);

    switch (message.type) {
      case "llm_response_chunk":
        appendMessage("ai", message.text_chunk || "");
        break;
      case "llm_response_complete":
        hideThinking();
        break;
      case "live_summary":
        appendMessage("system", `📊 Summary: ${message.text}`);
        break;
      case "ocr_result":
      case "transcript":
        appendMessage("system", `📝 ${message.text}`);
        if (message.type === "transcript") scheduleAutoCoach(message.text);
        break;
      case "error":
        hideThinking();
        appendMessage("system", "❌ Sorry, something went wrong.");
        break;
      default:
        console.warn("Unhandled message type:", message);
    }
  };

  socket.onclose = () => {
    console.log("🔌 Disconnected from backend");
    backendConnected = false;
    updateUIStatus();
    showNotification("Backend disconnected - retrying...", "warning");
    appendMessage("system", "🔌 Disconnected — retrying…");

    const delay = Math.min(10000, 1000 * Math.pow(2, reconnectAttempts));
    reconnectAttempts++;
    setTimeout(setupSocket, delay);
  };

  socket.onerror = () => {
    console.error("WebSocket error");
    backendConnected = false;
    updateUIStatus();
  };
}

setupSocket();

// --- Element Refs ---
const chatUI = document.getElementById("chat-ui");
const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-ui-input");
const chatSend = document.getElementById("chat-ui-send");
const closeBtn = document.getElementById("close-btn");
const copyBtn = document.getElementById("copy-btn");

const thinkingBar = document.getElementById("thinking-bar");
const thinkingText = document.getElementById("thinking-text");
const closeThinkingBtn = document.querySelector(".close-icon");

// --- Ask Button: Toggle Chat UI ---
document.getElementById("ask-btn")?.addEventListener("click", () => {
  chatUI.hidden = !chatUI.hidden;
  if (!chatUI.hidden) chatInput.focus();
});

// --- Toggle Keyboard Shortcut (⌘ or Ctrl + \) ---
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "\\") {
    chatUI.hidden = !chatUI.hidden;
    if (!chatUI.hidden) chatInput.focus();
  }
});

// --- Chat Input Handlers ---
chatSend?.addEventListener("click", sendChatMessage);
chatInput?.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendChatMessage();
  }
});

function sendChatMessage() {
  const text = chatInput.value.trim();
  if (!text || socket.readyState !== WebSocket.OPEN) return;

  chatInput.value = "";
  showThinking(text);
  appendMessage("user", text);

  socket.send(
    JSON.stringify({
      type: "llm_query",
      query: text,
      style: answerStyle,
    })
  );
}

function appendMessage(sender, text) {
  const msg = document.createElement("div");
  msg.className =
    "message " +
    (sender === "user" ? "user" : sender === "ai" ? "ai" : "system");
  msg.textContent = text;
  chatMessages.appendChild(msg);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// --- Thinking Display ---
function showThinking(text = "Cue is typing…") {
  const thinkingBar = document.getElementById("thinking-bar");
  const thinkingText = document.getElementById("thinking-text");
  if (thinkingText) thinkingText.textContent = text;
  if (thinkingBar) thinkingBar.classList.remove("hidden");
}

function hideThinking() {
  const thinkingBar = document.getElementById("thinking-bar");
  if (thinkingBar) thinkingBar.classList.add("hidden");
}

// --- Close & Copy ---
closeBtn?.addEventListener("click", () => (chatUI.hidden = true));
copyBtn?.addEventListener("click", () => {
  const transcript = [...chatMessages.children]
    .map((el) => el.textContent)
    .join("\n");
  navigator.clipboard.writeText(transcript);
});

// --- IPC Renderer Integration ---
const { ipcRenderer } = window.electronAPI || {};

ipcRenderer?.onBackendStatus?.((status) => {
  backendConnected = status.connected;
  updateUIStatus();
});

ipcRenderer?.onBackendMessage?.((msg) => {
  appendMessage("system", `📡 ${JSON.stringify(msg)}`);
});

// Button bindings
document.getElementById("listen-btn")?.addEventListener("click", () => {
  console.log("🎤 Listen clicked");
  ipcRenderer?.onToggleAudioCapture?.(() => {});
});
document.getElementById("monitor-btn")?.addEventListener("click", () => {
  console.log("🖥 Monitor clicked");
  ipcRenderer?.onToggleScreenCapture?.(() => {});
});
document.getElementById("stealth-btn")?.addEventListener("click", () => {
  console.log("🥷 Stealth clicked");
  window.electronAPI?.stealth?.toggle();
});

// --- Auto-Coach Logic ---
let autoCoachActive = false;
let lastAutoCoachAt = 0;
let autoCoachTimer = null;
let lastTranscriptSnippet = "";

function scheduleAutoCoach(text) {
  if (!autoCoachActive) return;
  lastTranscriptSnippet = text || "";
  if (autoCoachTimer) clearTimeout(autoCoachTimer);
  autoCoachTimer = setTimeout(() => tryAutoCoach(lastTranscriptSnippet), 1200);
}

function tryAutoCoach(text) {
  const now = Date.now();
  if (now - lastAutoCoachAt < 10000) return;
  if (!isLikelyQuestion(text)) return;
  if (socket.readyState !== WebSocket.OPEN) return;

  lastAutoCoachAt = now;
  const question = extractQuestion(text);
  if (!question) return;

  showThinking(question);
  appendMessage("system", "Thinking…");
  socket.send(
    JSON.stringify({
      type: "llm_query",
      query: question,
      trigger: "auto",
      style: answerStyle,
    })
  );
}

function isLikelyQuestion(text) {
  if (!text) return false;
  const t = text.trim();
  if (t.endsWith("?")) return true;
  const lower = t.toLowerCase();
  return [
    "how",
    "why",
    "what",
    "when",
    "where",
    "which",
    "could you",
    "can you",
    "would you",
    "do you",
    "tell me",
    "walk me",
    "explain",
  ].some((w) => lower.startsWith(w) || lower.includes(w + " "));
}
function extractQuestion(text) {
  const t = (text || "").trim();
  const qMatch = t.match(/[^?.!]*\?\s*$/);
  if (qMatch) return qMatch[0].trim();
  const parts = t
    .split(/[.!]/)
    .map((s) => s.trim())
    .filter(Boolean);
  return parts.length ? parts[parts.length - 1] : "";
}

// --- UI Init ---
document.addEventListener("DOMContentLoaded", () => updateUIStatus());
window.addEventListener("load", () => {
  updateUIStatus();
  checkBackendStatus();
});

// --- Backend status monitoring ---
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
    console.error("Failed to check backend status:", error);
  }
}
