// src/services/route.ts
let ws: WebSocket | null = null;
let listeners: ((msg: any) => void)[] = [];

export function connect() {
  if (ws && ws.readyState === WebSocket.OPEN) return;

  ws = new WebSocket("ws://127.0.0.1:8001/ws");

  ws.onopen = () => {
    console.log("✅ Connected to Cue backend");
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      listeners.forEach((cb) => cb(msg));
    } catch (err) {
      console.error("❌ Failed to parse message:", err);
    }
  };

  ws.onclose = () => {
    console.log("❌ Disconnected from Cue backend");
    ws = null;
  };
}

export function onMessage(cb: (msg: any) => void) {
  listeners.push(cb);
  return () => {
    listeners = listeners.filter((fn) => fn !== cb);
  };
}

export function sendMessage(payload: object) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(payload));
  } else {
    console.warn("⚠️ WebSocket not connected");
  }
}

// Session helpers
export function startSession() {
  sendMessage({ type: "start_session" });
}

export function stopSession() {
  sendMessage({ type: "stop_session" });
}

export function sendQuery(query: string, style?: string) {
  sendMessage({ type: "llm_query", query, style });
}
