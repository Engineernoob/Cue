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
let sessionActive = false;
let mediaRecorder = null;

document.getElementById("listen-btn")?.addEventListener("click", async () => {
  if (!sessionActive) {
    sessionActive = true;
    updateListenButton();
    await startAudioCapture();
  } else {
    sessionActive = false;
    updateListenButton();
    stopAudioCapture();
  }
});

function updateListenButton() {
  const btn = document.getElementById("listen-btn");
  if (btn) {
    btn.textContent = sessionActive ? "Pause ⏸" : "Listen ▶️";
  }
}

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
function showResponse(content, status = "Cue’s answer:") {
  if (responseText) responseText.textContent = content;
  if (responseStatus) responseStatus.textContent = status;
  if (responseBox) responseBox.hidden = false;
}

closeBtn?.addEventListener("click", () => {
  if (responseBox) responseBox.hidden = true;
});

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
  responseBox.hidden = true;
});

document.getElementById("menu-btn")?.addEventListener("click", () => {
  ipcRenderer?.send("open-menu");
});

document.getElementById("toggle-btn")?.addEventListener("click", () => {
  ipcRenderer?.send("toggle-overlay");
});
