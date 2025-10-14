import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import CueBar from "./components/Cuebar";
import CuePanel from "./components/CuePanel";
import StealthOverlay from "./components/StealthOverlay";
import {
  connect,
  onMessage,
  sendQuery,
  startSession,
  stopSession,
  BackendEvent,
} from "./services/socket";

export type CueMessage = {
  id: string;
  type: "transcript" | "screen" | "ai" | "auto_hint" | "system";
  text: string;
  streaming: boolean;
  timestamp: number;
};

const PANEL_STORAGE_KEY = "cue:panel-visible";
const HINTS_STORAGE_KEY = "cue:hints-enabled";
export const OVERLAY_IDLE_HIDE_MS = 6000;
export const OVERLAY_STREAMING_IDLE_HIDE_MS = 9000;

const readBoolean = (key: string, fallback: boolean) => {
  if (typeof window === "undefined") return fallback;
  const value = window.localStorage.getItem(key);
  if (value === null) return fallback;
  return value === "true";
};

function App() {
  const [showPanel, setShowPanel] = useState(() => readBoolean(PANEL_STORAGE_KEY, true));
  const [messages, setMessages] = useState<CueMessage[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<
    "connecting" | "connected" | "disconnected"
  >("connecting");
  const [sessionActive, setSessionActive] = useState(false);
  const [whisperLoaded, setWhisperLoaded] = useState(false);
  const [lastHeartbeat, setLastHeartbeat] = useState<number | null>(null);
  const [hintsEnabled, setHintsEnabled] = useState(() => readBoolean(HINTS_STORAGE_KEY, true));
  const [overlayVisible, setOverlayVisible] = useState(true);

  const hintsRef = useRef(hintsEnabled);
  const overlayTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(PANEL_STORAGE_KEY, String(showPanel));
  }, [showPanel]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(HINTS_STORAGE_KEY, String(hintsEnabled));
    hintsRef.current = hintsEnabled;
  }, [hintsEnabled]);

  const scheduleOverlayHide = useCallback(
    (delay: number) => {
      if (typeof window === "undefined") return;
      if (overlayTimerRef.current) {
        window.clearTimeout(overlayTimerRef.current);
      }
      overlayTimerRef.current = window.setTimeout(() => {
        setOverlayVisible(false);
        overlayTimerRef.current = null;
      }, delay);
    },
    [],
  );

  const registerOverlayActivity = useCallback(
    (streaming: boolean) => {
      setOverlayVisible(true);
      scheduleOverlayHide(
        streaming ? OVERLAY_STREAMING_IDLE_HIDE_MS : OVERLAY_IDLE_HIDE_MS,
      );
    },
    [scheduleOverlayHide],
  );

  const appendMessage = useCallback(
    (id: string, type: CueMessage["type"], text: string, streaming: boolean) => {
      if (!text.trim()) return;
      setMessages((prev) => {
        const index = prev.findIndex((message) => message.id === id);
        if (index === -1) {
          return [
            ...prev,
            {
              id,
              type,
              text,
              streaming,
              timestamp: Date.now(),
            },
          ];
        }

        const next = [...prev];
        next[index] = {
          ...next[index],
          text,
          streaming,
        };
        return next;
      });
    },
    [],
  );

  const pushSystemMessage = useCallback(
    (text: string) => {
      const id = `system-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
      appendMessage(id, "system", text, false);
    },
    [appendMessage],
  );

  const handleEvent = useCallback(
    (event: BackendEvent) => {
      switch (event.type) {
        case "connection":
          setConnectionStatus(event.status);
          return;
        case "transcript":
          appendMessage(event.id, "transcript", event.text, false);
          registerOverlayActivity(false);
          return;
        case "ocr":
          appendMessage(event.id, "screen", event.text, false);
          return;
        case "llm":
          appendMessage(event.id, "ai", event.text, event.streaming);
          registerOverlayActivity(event.streaming);
          return;
        case "hint":
          if (!hintsRef.current) return;
          appendMessage(event.id, "auto_hint", event.text, event.streaming);
          registerOverlayActivity(event.streaming);
          return;
        case "session":
          setSessionActive(event.status === "started");
          pushSystemMessage(
            event.status === "started" ? "Session started." : "Session stopped.",
          );
          registerOverlayActivity(false);
          return;
        case "heartbeat":
          setLastHeartbeat(event.timestamp);
          setWhisperLoaded(event.whisperLoaded);
          setSessionActive(event.sessionActive);
          return;
        case "error":
          pushSystemMessage(`${event.scope}: ${event.message}`);
          registerOverlayActivity(false);
          return;
      }
    },
    [appendMessage, pushSystemMessage, registerOverlayActivity],
  );

  useEffect(() => {
    connect();

    const unsubscribe = onMessage(handleEvent);

    return () => unsubscribe();
  }, [handleEvent]);

  useEffect(() => {
    scheduleOverlayHide(OVERLAY_IDLE_HIDE_MS);
    return () => {
      if (typeof window !== "undefined" && overlayTimerRef.current) {
        window.clearTimeout(overlayTimerRef.current);
      }
    };
  }, [scheduleOverlayHide]);

  const heartbeatLabel = useMemo(() => {
    if (!lastHeartbeat) return "Awaiting heartbeat…";
    const secondsAgo = Math.max(0, Math.round(Date.now() / 1000 - lastHeartbeat));
    if (secondsAgo < 5) return "Heartbeat: just now";
    if (secondsAgo < 60) return `Heartbeat: ${secondsAgo}s ago`;
    const minutes = Math.floor(secondsAgo / 60);
    return `Heartbeat: ${minutes}m ago`;
  }, [lastHeartbeat]);

  const contextItems = [
    {
      label: "Connection",
      value: connectionStatus === "connected" ? "Connected" : connectionStatus,
      tone:
        connectionStatus === "connected"
          ? "bg-emerald-400/20 text-emerald-200"
          : connectionStatus === "connecting"
          ? "bg-amber-400/20 text-amber-100"
          : "bg-rose-500/20 text-rose-100",
    },
    {
      label: "Session",
      value: sessionActive ? "Live" : "Idle",
      tone: sessionActive ? "bg-sky-500/20 text-sky-100" : "bg-white/10 text-slate-200",
    },
    {
      label: "Whisper",
      value: whisperLoaded ? "Ready" : "Loading",
      tone: whisperLoaded ? "bg-indigo-500/25 text-indigo-100" : "bg-white/10 text-slate-200",
    },
    {
      label: "Heartbeat",
      value: heartbeatLabel.replace("Heartbeat: ", ""),
      tone: "bg-white/10 text-slate-200",
    },
  ];

  return (
    <StealthOverlay visible={overlayVisible}>
      <header className="flex items-center justify-between rounded-3xl border border-white/10 bg-white/5 px-6 py-4 shadow-[0_20px_60px_rgba(10,21,55,0.35)] backdrop-blur">
        <div className="flex flex-col">
          <span className="text-xs uppercase tracking-[0.28em] text-slate-200/60">
            Cue overlay
          </span>
          <h1 className="text-lg font-semibold text-white">Stealth Auto-Fade engaged</h1>
        </div>
        <div className="flex flex-wrap justify-end gap-2">
          {contextItems.map((item) => (
            <span
              key={item.label}
              className={`flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium transition duration-200 ${item.tone}`}
            >
              <span className="uppercase tracking-wide text-[10px] text-white/70">
                {item.label}
              </span>
              <span className="text-sm capitalize text-white">{item.value}</span>
            </span>
          ))}
        </div>
      </header>

      <main className="flex flex-1 flex-col">
        <CuePanel
          show={showPanel}
          messages={messages}
          hintsEnabled={hintsEnabled}
          onToggleHints={setHintsEnabled}
          heartbeatLabel={heartbeatLabel}
        />
      </main>

      <CueBar
        showPanel={showPanel}
        setShowPanel={setShowPanel}
        onStart={startSession}
        onStop={stopSession}
        onSendQuery={sendQuery}
        connectionStatus={connectionStatus}
        sessionActive={sessionActive}
        whisperLoaded={whisperLoaded}
      />
    </StealthOverlay>
  );
}

export default App;
