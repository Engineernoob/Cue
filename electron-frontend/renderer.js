const { ipcRenderer } = window.electron || {};

// --- Element Refs ---
const chatWindow = document.getElementById("chat-window");
const chatBox = document.getElementById("chat-input-box");
const sendBtn = document.getElementById("chat-send-btn");
const chatMessages = document.getElementById("chat-messages");

const inputBar = document.getElementById("chat-input");
const thinkingBar = document.getElementById("thinking-bar");
const thinkingText = document.getElementById("thinking-text");
const closeThinkingBtn = document.querySelector(".close-icon");
const insightsText = document.getElementById("insights-content");

let sessionActive = false;
let mediaRecorder = null;

// --- Audio Capture Functions ---
async function startAudioCapture() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        const reader = new FileReader();
        reader.onload = () => {
          // Convert audio blob to base64 string and send to backend
          const base64data = reader.result.split(",")[1]; // Strip off data URL prefix
          ipcRenderer.send("audio-chunk-data", base64data);
        };
        reader.readAsDataURL(event.data); // Read as data URL for base64 encoding
      }
    };

    mediaRecorder.start(250); // emit audio chunks every 250ms
  } catch (err) {
    console.error("Error accessing microphone:", err);
  }
}

function stopAudioCapture() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    mediaRecorder = null;
  }
}

// --- Listen Button Handler ---
document.getElementById("listen-btn")?.addEventListener("click", async () => {
  if (!sessionActive) {
    try {
      await startSession("default");
      sessionActive = true;
      updateListenButton();
      await startAudioCapture(); // start capturing mic audio
    } catch (e) {
      console.error("Error starting session or audio capture:", e);
    }
  } else {
    try {
      await stopSession();
      sessionActive = false;
      updateListenButton();
      stopAudioCapture(); // stop mic audio capture
    } catch (e) {
      console.error("Error stopping session or audio capture:", e);
    }
  }
});

function updateListenButton() {
  const btn = document.getElementById("listen-btn");
  if (!btn) return;
  btn.textContent = sessionActive ? "Pause ⏸" : "Listen ▶️";
}

// --- Send Chat Message (Ask button or Enter key) ---
function sendChatMessage() {
  const text = chatBox.value.trim();
  if (!text) return;
  appendMessage(text, "user");
  showThinking(text);
  ipcRenderer?.send("send-llm-query", text);
  chatBox.value = "";
}

document
  .getElementById("chat-send-btn")
  ?.addEventListener("click", sendChatMessage);

chatBox?.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendChatMessage();
  }
});

// Toggle chat with Cmd+\ or Ctrl+\
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "\\") {
    if (chatWindow) {
      chatWindow.hidden = !chatWindow.hidden;
      if (!chatWindow.hidden) chatBox?.focus();
    }
  }
});

// --- Smart Text Parsing ---
function processIncomingText(text) {
  const actionRegex =
    /(TODO|Action Item|Follow up|Reminder|Next step):?\s*(.*)/gi;
  const questionRegex =
    /\b(how|what|when|where|why|can|should|does)\b[^.?!]*[.?!]/gi;

  const actionMatches = [...text.matchAll(actionRegex)];
  const questionMatches = [...text.matchAll(questionRegex)];

  actionMatches.forEach((match) => {
    console.log("🔧 Action Item:", match[2]);
  });

  questionMatches.forEach((match) => {
    const question = match[0].trim();
    showThinking(question);
    console.log("🧠 Auto-Query:", question);
    ipcRenderer?.send("send-llm-query", question);
    appendMessage(question, "user");
  });
}

// --- Chat UI ---
function appendMessage(text, role = "user") {
  if (!chatMessages) return;

  const msg = document.createElement("div");
  msg.className = `chat-message ${role}`;
  msg.textContent = text;
  msg.style.textAlign = role === "user" ? "right" : "left";
  msg.style.opacity = role === "system" ? 0.7 : 1;
  msg.style.marginBottom = "8px";
  chatMessages.appendChild(msg);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// --- Thinking Bar ---
function showThinking(text = "") {
  if (!thinkingBar) return;
  thinkingText.textContent = `“${text}”`;
  thinkingBar.hidden = false;
  thinkingBar.classList.add("show");
}

function hideThinking() {
  thinkingBar?.classList.remove("show");
  setTimeout(() => {
    if (thinkingBar) thinkingBar.hidden = true;
  }, 600);
}

closeThinkingBtn?.addEventListener("click", hideThinking);

// --- Insights Controls ---
document.getElementById("live-insights-btn")?.addEventListener("click", () => {
  alert("Live Insights feature coming soon!");
});

document
  .getElementById("show-transcript-btn")
  ?.addEventListener("click", () => {
    if (chatWindow) {
      chatWindow.hidden = !chatWindow.hidden;
      if (!chatWindow.hidden) chatBox?.focus();
    }
  });

// --- IPC Listeners ---
ipcRenderer?.on("backend-status", (_e, status) => {
  console.log(
    "[Backend Status]",
    status.connected ? "Connected" : "Disconnected"
  );
});

ipcRenderer?.on("backend-message", (_e, message) => {
  console.log("[From Backend]", message);

  switch (message.type) {
    case "transcript":
    case "ocr_result":
      processIncomingText(message.text);
      break;
    case "llm_response_chunk":
      appendMessage(message.text_chunk || "", "assistant");
      break;
    case "llm_response_complete":
      appendMessage(message.text || "", "assistant");
      hideThinking();
      break;
    case "live_summary":
      if (insightsText) insightsText.textContent = message.text;
      break;
    case "transcript_error":
    case "ocr_error":
    case "llm_response_error":
      console.error("[Error]", message.message);
      hideThinking();
      break;
    default:
      console.warn("[Unhandled]", message.type);
  }
});

ipcRenderer?.on("llm-response-error", (_e, error) => {
  console.error("[LLM Error]", error.message);
  hideThinking();
});

ipcRenderer.on("show-coaching-prompt", (_e, prompt) => {
  const box = document.getElementById("prompt-box");
  if (box) {
    box.innerText = prompt;
    box.hidden = false;
    setTimeout(() => (box.hidden = true), 6000);
  }
});

// --- Manual Coaching Prompt ---
async function showPrompt(type = "default") {
  const box = document.getElementById("prompt-box");
  try {
    const prompt = await ipcRenderer.invoke("coaching:get-prompt", type);
    if (box) {
      box.innerText = prompt;
      box.hidden = false;
      setTimeout(() => (box.hidden = true), 6000);
    }
  } catch (error) {
    console.error("Failed to get coaching prompt:", error);
  }
}

// --- Session Controls ---
async function startSession(sessionType = "default") {
  try {
    const result = await ipcRenderer.invoke("start-session", {
      platform: "generic",
      problemTitle: sessionType,
    });
    console.log("Session started:", result);
  } catch (err) {
    console.error("Failed to start session:", err);
    throw err;
  }
}

async function stopSession() {
  try {
    const result = await ipcRenderer.invoke("stop-session");
    console.log("Session stopped:", result);
  } catch (err) {
    console.error("Failed to stop session:", err);
    throw err;
  }
}

// --- Expose session control & sendChatMessage ---
window.startSession = startSession;
window.stopSession = stopSession;
window.sendChatMessage = sendChatMessage;
