// renderer.js (for Cue floating overlay)
const { ipcRenderer } = window.electron || {};

// --- Button Handlers (from HTML) ---
document.getElementById("listen-btn")?.addEventListener("click", () => {
  ipcRenderer?.send("toggle-audio-capture");
});

document.getElementById("ask-btn")?.addEventListener("click", () => {
  showPrompt("interview");
});

document.getElementById("toggle-btn")?.addEventListener("click", () => {
  ipcRenderer?.send("request-hide-window");
});

document.getElementById("menu-btn")?.addEventListener("click", () => {
  alert("More options coming soon.");
});

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
  }
}

async function stopSession() {
  try {
    const result = await ipcRenderer.invoke("stop-session");
    console.log("Session stopped:", result);
  } catch (err) {
    console.error("Failed to stop session:", err);
  }
}

// --- Smart Text Parsing ---
function processIncomingText(text) {
  const actionRegex = /(TODO|Action Item|Follow up|Reminder|Next step):?\s*(.*)/gi;
  const questionRegex = /\b(how|what|when|where|why|can|should|does)\b[^.?!]*[.?!]/gi;

  const actionMatches = [...text.matchAll(actionRegex)];
  const questionMatches = [...text.matchAll(questionRegex)];

  actionMatches.forEach((match) => {
    console.log("🔧 Action Item:", match[2]);
  });

  questionMatches.forEach((match) => {
    const question = match[0].trim();
    console.log("🧠 Auto-Query:", question);
    ipcRenderer?.send("send-llm-query", question);
    appendMessage(question, "user");
  });
}

// --- Chat UI ---
const chatWindow = document.getElementById("chat-window");
const chatBox = document.getElementById("chat-input-box");
const sendBtn = document.getElementById("chat-send-btn");
const chatMessages = document.getElementById("chat-messages");

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

function sendChatMessage() {
  const text = chatBox.value.trim();
  if (!text) return;
  appendMessage(text, "user");
  ipcRenderer?.send("send-llm-query", text);
  chatBox.value = "";
}

sendBtn?.addEventListener("click", sendChatMessage);

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

// --- IPC Listeners ---
ipcRenderer?.on("backend-status", (_e, status) => {
  console.log("[Backend Status]", status.connected ? "Connected" : "Disconnected");
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
      break;
    case "transcript_error":
    case "ocr_error":
    case "llm_response_error":
      console.error("[Error]", message.message);
      break;
    default:
      console.warn("[Unhandled]", message.type);
  }
});

ipcRenderer?.on("llm-response-error", (_e, error) => {
  console.error("[LLM Error]", error.message);
});

// Show coaching prompts
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

// --- Expose session control ---
window.startSession = startSession;
window.stopSession = stopSession;