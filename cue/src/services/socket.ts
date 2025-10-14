// src/services/socket.ts

type BackendEvent =
  | { type: "transcript"; id: string; text: string }
  | { type: "ocr"; id: string; text: string }
  | { type: "llm"; id: string; text: string; streaming: boolean }
  | { type: "hint"; id: string; text: string; streaming: boolean }
  | { type: "session"; status: "started" | "stopped" }
  | { type: "heartbeat"; timestamp: number; whisperLoaded: boolean; sessionActive: boolean }
  | { type: "connection"; status: "connecting" | "connected" | "disconnected" }
  | { type: "error"; scope: string; message: string };

const listeners = new Set<(event: BackendEvent) => void>();

let ws: WebSocket | null = null;
let reconnectAttempts = 0;
let heartbeatTimer: ReturnType<typeof setTimeout> | null = null;
let lastHeartbeat = 0;

const WS_ENDPOINT = "ws://127.0.0.1:8001/ws";
const HEARTBEAT_GRACE_MS = 45_000;
const RECONNECT_BASE_MS = 1_000;
const RECONNECT_MAX_MS = 10_000;

const llmStreams = new Map<string, string>();
const hintStreams = new Map<string, string>();

const emit = (event: BackendEvent) => {
  listeners.forEach((listener) => listener(event));
};

const resetHeartbeatTimer = () => {
  if (heartbeatTimer) {
    clearTimeout(heartbeatTimer);
  }
  heartbeatTimer = setTimeout(() => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const elapsed = Date.now() - lastHeartbeat;
    if (elapsed > HEARTBEAT_GRACE_MS) {
      console.warn("⚠️ Heartbeat timeout, reconnecting...");
      ws.close();
    }
  }, HEARTBEAT_GRACE_MS);
};

const scheduleReconnect = () => {
  reconnectAttempts += 1;
  const delay = Math.min(
    RECONNECT_MAX_MS,
    RECONNECT_BASE_MS * Math.pow(2, reconnectAttempts - 1),
  );
  emit({ type: "connection", status: "connecting" });
  setTimeout(() => connect(true), delay);
};

type StreamPayload =
  | { status: string; text?: string; response_id: string; hint_id?: undefined }
  | { status: string; text?: string; hint_id: string; response_id?: undefined };

const handleStreamEvent = (payload: StreamPayload, kind: "llm_response" | "auto_hint") => {
  const map = kind === "llm_response" ? llmStreams : hintStreams;
  const id = "response_id" in payload ? payload.response_id : payload.hint_id;
  if (!id) return;

  if (payload.status === "chunk" && payload.text) {
    const next = (map.get(id) ?? "") + payload.text;
    map.set(id, next);
    emit({
      type: kind === "llm_response" ? "llm" : "hint",
      id,
      text: next,
      streaming: true,
    });
    return;
  }

  if (payload.status === "complete") {
    const finalText = (map.get(id) ?? "") || payload.text || "";
    map.delete(id);
    if (!finalText) return;
    emit({
      type: kind === "llm_response" ? "llm" : "hint",
      id,
      text: finalText,
      streaming: false,
    });
  }
};

const handleIncoming = (raw: unknown) => {
  if (!raw || typeof raw !== "object") return;
  const data = raw as Record<string, unknown>;
  const asString = (value: unknown): value is string => typeof value === "string" && value.length > 0;
  const asStatus = (value: unknown): value is string => typeof value === "string";
  switch (data.type) {
    case "transcript":
      if (asString(data.text) && asString(data.message_id)) {
        emit({ type: "transcript", id: data.message_id, text: data.text });
      }
      return;
    case "ocr_result":
      if (asString(data.text) && asString(data.message_id)) {
        emit({ type: "ocr", id: data.message_id, text: data.text });
      }
      return;
    case "llm_response":
      if (asStatus(data.status) && asString(data.response_id)) {
        handleStreamEvent(
          {
            status: data.status,
            text: asString(data.text) ? data.text : undefined,
            response_id: data.response_id,
          },
          "llm_response",
        );
      }
      return;
    case "auto_hint":
      if (asStatus(data.status) && asString(data.hint_id)) {
        handleStreamEvent(
          {
            status: data.status,
            text: asString(data.text) ? data.text : undefined,
            hint_id: data.hint_id,
          },
          "auto_hint",
        );
      }
      return;
    case "session_status":
      if (data.status === "started" || data.status === "stopped") {
        emit({ type: "session", status: data.status });
      }
      return;
    case "heartbeat":
      lastHeartbeat = Date.now();
      resetHeartbeatTimer();
      emit({
        type: "heartbeat",
        timestamp: typeof data.timestamp === "number" ? data.timestamp : Date.now() / 1000,
        whisperLoaded: Boolean(data.whisper_loaded),
        sessionActive: Boolean(data.session_active),
      });
      return;
    case "llm_response_error":
    case "auto_hint_error":
    case "transcript_error":
    case "ocr_error":
    case "error":
      emit({
        type: "error",
        scope: String(data.type ?? "unknown"),
        message: typeof data.message === "string" ? data.message : "Unexpected error",
      });
      return;
    default:
      return;
  }
};

export function connect(force = false) {
  if (!force && ws && ws.readyState === WebSocket.OPEN) return;
  if (ws && ws.readyState === WebSocket.CONNECTING) return;

  emit({ type: "connection", status: "connecting" });
  ws = new WebSocket(WS_ENDPOINT);

  ws.onopen = () => {
    reconnectAttempts = 0;
    lastHeartbeat = Date.now();
    resetHeartbeatTimer();
    console.info("✅ Connected to Cue backend");
    emit({ type: "connection", status: "connected" });
  };

  ws.onmessage = (event) => {
    try {
      if (typeof event.data !== "string") {
        console.warn("⚠️ Received non-text WebSocket payload");
        return;
      }
      const parsed: unknown = JSON.parse(event.data);
      handleIncoming(parsed);
    } catch (error) {
      console.error("❌ Failed to parse message:", error);
    }
  };

  ws.onerror = (event) => {
    console.error("❌ WebSocket error", event);
  };

  ws.onclose = () => {
    console.warn("⚠️ Disconnected from Cue backend");
    ws = null;
    if (heartbeatTimer) {
      clearTimeout(heartbeatTimer);
      heartbeatTimer = null;
    }
    llmStreams.clear();
    hintStreams.clear();
    emit({ type: "connection", status: "disconnected" });
    scheduleReconnect();
  };
}

export function onMessage(listener: (event: BackendEvent) => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function sendMessage(payload: object) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(payload));
  } else {
    console.warn("⚠️ WebSocket not connected");
  }
}

export function startSession() {
  sendMessage({ type: "start_session" });
}

export function stopSession() {
  sendMessage({ type: "stop_session" });
}

export function sendQuery(query: string, style?: string) {
  sendMessage({ type: "llm_query", query, style });
}

export type { BackendEvent };
