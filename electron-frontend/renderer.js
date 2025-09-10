// --- WebSocket Setup ---
let socket = null;
let reconnectAttempts = 0;

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
        if (message.type === "transcript") {
          scheduleAutoCoach(message.text);
        }
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

    // Reconnect with exponential backoff
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
const inputBar = document.getElementById("input-bar");
const inputBox = document.getElementById("chat-input-box");
const sendBtn = document.getElementById("chat-send-btn");

const thinkingBar = document.getElementById("thinking-bar");
const thinkingText = document.getElementById("thinking-text");
const closeThinkingBtn = document.querySelector(".close-icon");

// Unified Chat UI
const chatUI = document.getElementById("chat-ui");
const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-ui-input");
const chatSend = document.getElementById("chat-ui-send");
const closeBtn = document.getElementById("close-btn");
const copyBtn = document.getElementById("copy-btn");

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

// --- Audio Capture Logic (Float32 PCM for Whisper) ---
let audioContext = null;
let mediaStream = null;
let scriptNode = null;
let silentGain = null;

let noiseFloor = 0.005;
let smoothing = 0.95;

function float32ToBase64(float32Array) {
  const buffer = new ArrayBuffer(float32Array.length * 4);
  const view = new DataView(buffer);
  for (let i = 0; i < float32Array.length; i++) {
    view.setFloat32(i * 4, float32Array[i], true);
  }
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function resampleTo16kHz(input, inputSampleRate) {
  const targetRate = 16000;
  if (inputSampleRate === targetRate) return input;
  const ratio = inputSampleRate / targetRate;
  const newLength = Math.round(input.length / ratio);
  const output = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const idx = i * ratio;
    const idx1 = Math.floor(idx);
    const idx2 = Math.min(idx1 + 1, input.length - 1);
    const frac = idx - idx1;
    output[i] = input[idx1] * (1 - frac) + input[idx2] * frac;
  }
  return output;
}

function isSilentAdaptive(float32Array, multiplier = 3.0) {
  let sum = 0;
  for (let i = 0; i < float32Array.length; i++) {
    sum += Math.abs(float32Array[i]);
  }
  const avg = sum / float32Array.length;
  noiseFloor = smoothing * noiseFloor + (1 - smoothing) * avg;
  return avg < noiseFloor * multiplier;
}

async function startAudioCapture() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(mediaStream);

    const bufferSize = 4096;
    scriptNode = audioContext.createScriptProcessor(bufferSize, 1, 1);
    scriptNode.onaudioprocess = (e) => {
      if (socket.readyState !== WebSocket.OPEN) return;
      const input = e.inputBuffer.getChannelData(0);
      if (isSilentAdaptive(input)) return;

      const pcm16k = resampleTo16kHz(input, audioContext.sampleRate);
      const b64 = float32ToBase64(pcm16k);
      socket.send(JSON.stringify({ type: "audio_chunk", data: b64 }));
    };

    source.connect(scriptNode);
    silentGain = audioContext.createGain();
    silentGain.gain.value = 0;
    scriptNode.connect(silentGain);
    silentGain.connect(audioContext.destination);
  } catch (err) {
    console.error("Microphone error:", err);
  }
}

function stopAudioCapture() {
  try {
    if (scriptNode) {
      scriptNode.disconnect();
      scriptNode.onaudioprocess = null;
      scriptNode = null;
    }
    if (silentGain) {
      try {
        silentGain.disconnect();
      } catch {}
      silentGain = null;
    }
    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }
    if (mediaStream) {
      for (const track of mediaStream.getTracks()) track.stop();
      mediaStream = null;
    }
  } catch (e) {}
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

// --- Close & Copy ---
closeBtn?.addEventListener("click", () => (chatUI.hidden = true));
copyBtn?.addEventListener("click", () => {
  const transcript = [...chatMessages.children]
    .map((el) => el.textContent)
    .join("\n");
  navigator.clipboard.writeText(transcript);
});

// --- IPC Renderer Integration ---
const { ipcRenderer } = window.Electron || {};

ipcRenderer?.on("show-input", () => {
  chatUI.hidden = false;
  chatInput.focus();
});
ipcRenderer?.on("hide-response", () => {
  chatUI.hidden = true;
});

// --- Auto-Coach Logic (kept as-is) ---
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
  const qWords = [
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
  ];
  return qWords.some((w) => lower.startsWith(w) || lower.includes(w + " "));
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

// Initialize UI
document.addEventListener("DOMContentLoaded", () => updateUIStatus());
window.addEventListener("load", () => {
  updateUIStatus();
  checkBackendStatus();
});

// Backend status monitoring
async function checkBackendStatus() {
  try {
    const status = await window.Electron?.backend?.getStatus();
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
