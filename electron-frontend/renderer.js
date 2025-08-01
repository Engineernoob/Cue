const { ipcRenderer } = window.electron || {};

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

let sessionActive = false;
let mediaRecorder = null;

// --- Audio Session Handling ---
document.getElementById("listen-btn")?.addEventListener("click", async () => {
  if (!sessionActive) {
    try {
      await startSession();
      sessionActive = true;
      updateListenButton();
      await startAudioCapture();
    } catch (e) {
      console.error("Failed to start session:", e);
    }
  } else {
    try {
      await stopSession();
      sessionActive = false;
      updateListenButton();
      stopAudioCapture();
    } catch (e) {
      console.error("Failed to stop session:", e);
    }
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

// --- Toggle (⌘/Ctrl + \) ---
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "\\") {
    inputBar.hidden = !inputBar.hidden;
    if (!inputBar.hidden) inputBox.focus();
  }
});

// --- Input Actions ---
sendBtn?.addEventListener("click", sendChatMessage);
inputBox?.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendChatMessage();
  }
});

function sendChatMessage() {
  const text = inputBox.value.trim();
  if (!text) return;

  inputBox.value = "";
  showThinking(text);
  showResponse("Waiting for response...", "Thinking…");
  ipcRenderer?.send("send-llm-query", text);
}

// --- Audio Capture ---
async function startAudioCapture() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        const reader = new FileReader();
        reader.onload = () => {
          const base64data = reader.result.split(",")[1];
          ipcRenderer.send("audio-chunk-data", base64data);
        };
        reader.readAsDataURL(event.data);
      }
    };
    mediaRecorder.start(250);
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

// --- Session IPC ---
async function startSession() {
  return ipcRenderer.invoke("start-session", {
    platform: "generic",
    problemTitle: "adhd-cue",
  });
}

async function stopSession() {
  return ipcRenderer.invoke("stop-session");
}

// --- Thinking Display ---
function showThinking(text = "") {
  thinkingText.textContent = `“${text}”`;
  thinkingBar.hidden = false;
  thinkingBar.classList.add("show");
}

function hideThinking() {
  thinkingBar.classList.remove("show");
  setTimeout(() => (thinkingBar.hidden = true), 600);
}

closeThinkingBtn?.addEventListener("click", hideThinking);

// --- Response Display ---
function showResponse(content, status = "Cue’s answer:") {
  responseText.textContent = content;
  responseStatus.textContent = status;
  responseBox.hidden = false;
}

closeBtn?.addEventListener("click", () => {
  responseBox.hidden = true;
});

copyBtn?.addEventListener("click", () => {
  navigator.clipboard.writeText(responseText.textContent.trim());
});

// --- Coaching Prompt Support ---
ipcRenderer.on("show-coaching-prompt", (_e, prompt) => {
  if (promptBox) {
    promptBox.innerText = prompt;
    promptBox.hidden = false;
    setTimeout(() => (promptBox.hidden = true), 6000);
  }
});

ipcRenderer.on("llm-response-error", (_e, err) => {
  console.error("[LLM Error]", err.message);
  hideThinking();
  showResponse("Sorry, something went wrong.", "Error");
});

// --- Streaming Response Handling ---
ipcRenderer.on("backend-message", (_e, msg) => {
  switch (msg.type) {
    case "llm_response_chunk":
      responseText.textContent += msg.text_chunk || "";
      break;
    case "llm_response_complete":
      hideThinking();
      break;
    case "live_summary":
      const el = document.getElementById("summary-text");
      if (el) el.textContent = msg.text;
      break;
    default:
      console.log("Unhandled message:", msg);
  }
});