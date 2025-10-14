import {
  useState,
  useRef,
  useEffect,
  Dispatch,
  SetStateAction,
  useCallback,
} from "react";
import { motion } from "framer-motion";
import { Pause, Mic, Brain, PanelRight, PanelLeft } from "lucide-react";
import RecordRTC from "recordrtc";
import { sendMessage } from "../services/socket";

interface Props {
  showPanel: boolean;
  setShowPanel: Dispatch<SetStateAction<boolean>>;
  onStart: () => void;
  onStop: () => void;
  onSendQuery: (query: string) => void;
  connectionStatus: "connecting" | "connected" | "disconnected";
  sessionActive: boolean;
  whisperLoaded: boolean;
}

const QUERY_STORAGE_KEY = "cue:query-draft";

export default function CueBar({
  showPanel,
  setShowPanel,
  onStart,
  onStop,
  onSendQuery,
  connectionStatus,
  sessionActive,
  whisperLoaded,
}: Props) {
  const [isRecording, setIsRecording] = useState(false);
  const [query, setQuery] = useState(() => {
    if (typeof window === "undefined") return "";
    return window.localStorage.getItem(QUERY_STORAGE_KEY) ?? "";
  });

  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<InstanceType<typeof RecordRTC> | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(QUERY_STORAGE_KEY, query);
  }, [query]);

  useEffect(() => {
    return () => {
      if (recorderRef.current) {
        recorderRef.current.stopRecording(() => undefined);
        recorderRef.current = null;
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  }, []);

  const toggleRecording = useCallback(async () => {
    if (!isRecording) {
      try {
        onStart();
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        streamRef.current = stream;
        const recorder = new RecordRTC(stream, {
          type: "audio",
          mimeType: "audio/wav",
          recorderType: RecordRTC.StereoAudioRecorder,
          timeSlice: 1000,
          desiredSampRate: 16000,
          numberOfAudioChannels: 1,
          ondataavailable: (blob: Blob) => {
            void blob.arrayBuffer().then((buffer) => {
              const bytes = new Uint8Array(buffer);
              const b64 = btoa(String.fromCharCode(...bytes));
              sendMessage({ type: "audio_chunk", data: b64 });
            });
          },
        });
        recorder.startRecording();
        recorderRef.current = recorder;
        setIsRecording(true);
      } catch (error) {
        console.error("Failed to access microphone:", error);
        onStop();
      }
      return;
    }

    if (recorderRef.current) {
      recorderRef.current.stopRecording(() => undefined);
      recorderRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    onStop();
    setIsRecording(false);
  }, [isRecording, onStart, onStop]);

  const handleToggleClick = () => {
    void toggleRecording();
  };

  const askAI = () => {
    if (!query.trim()) return;
    onSendQuery(query);
    setQuery("");
    setShowPanel(true);
  };

  return (
    <motion.div
      role="toolbar"
      aria-label="Cue controls"
      initial={{ opacity: 0, y: 32 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.28, ease: "easeOut" }}
      className="flex flex-col gap-3"
    >
      <div className="flex items-center gap-3 rounded-[36px] border border-white/10 bg-white/[0.08] px-6 py-4 text-white shadow-[0_20px_80px_rgba(10,21,55,0.45)] backdrop-blur-2xl">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => setShowPanel((prev) => !prev)}
            className="group flex items-center gap-2 rounded-full border border-white/15 bg-white/10 px-4 py-3 text-sm font-semibold uppercase tracking-wide transition-all duration-200 hover:-translate-y-0.5 hover:border-white/35 hover:bg-white/20"
            aria-label="Toggle Cue panel"
            aria-pressed={showPanel}
          >
            {showPanel ? (
              <PanelRight className="h-5 w-5 text-white transition-transform group-hover:scale-110" />
            ) : (
              <PanelLeft className="h-5 w-5 text-white transition-transform group-hover:scale-110" />
            )}
            <span>{showPanel ? "Hide" : "Show"}</span>
          </button>

          <button
            type="button"
            className={`group flex items-center gap-3 rounded-full px-5 py-3 text-sm font-semibold uppercase tracking-wide transition-all duration-200 hover:-translate-y-0.5 ${
              isRecording
                ? "border border-rose-300/40 bg-rose-400/20 text-rose-100 shadow-[0_12px_30px_rgba(244,63,94,0.35)]"
                : "border border-emerald-300/40 bg-emerald-400/15 text-emerald-100 shadow-[0_12px_30px_rgba(16,185,129,0.25)] hover:shadow-[0_18px_40px_rgba(16,185,129,0.35)]"
            }`}
            onClick={handleToggleClick}
            aria-label={isRecording ? "Stop recording" : "Start recording"}
            aria-pressed={isRecording}
          >
            {isRecording ? (
              <Pause className="h-5 w-5 transition-transform group-hover:scale-110" />
            ) : (
              <Mic className="h-5 w-5 transition-transform group-hover:scale-110" />
            )}
            <span>{isRecording ? "Listening" : "Listen"}</span>
          </button>
        </div>

        <div className="ml-auto flex flex-1 items-center gap-3 rounded-[24px] border border-white/10 bg-white/5 px-3 py-2">
          <label htmlFor="cue-query" className="sr-only">
            Ask Cue anything
          </label>
          <input
            id="cue-query"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                askAI();
              }
            }}
            placeholder="Type a prompt for Cue…"
            className="flex-1 rounded-2xl bg-transparent px-3 text-sm outline-none placeholder:text-slate-200/60"
          />
          <button
            type="button"
            className="group/button flex items-center gap-2 rounded-full bg-gradient-to-r from-blue-500 via-indigo-500 to-sky-400 px-5 py-2 text-sm font-semibold text-white shadow-[0_12px_32px_rgba(29,78,216,0.45)] transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_18px_42px_rgba(59,130,246,0.55)] disabled:translate-y-0 disabled:bg-white/20 disabled:text-slate-200/60 disabled:shadow-none"
            onClick={askAI}
            disabled={!query.trim() || connectionStatus !== "connected"}
          >
            <Brain className="h-4 w-4 transition-transform group-hover/button:scale-110" />
            <span>Ask AI</span>
          </button>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 px-2 text-xs text-slate-200/70" aria-live="polite" role="status">
        <span className="font-mono uppercase tracking-wide">
          {isRecording ? "Recording audio" : sessionActive ? "Session active" : "Session idle"}
        </span>
        <div className="flex flex-wrap items-center gap-3">
          <span className="flex items-center gap-1">
            <span
              className={`inline-flex h-2 w-2 rounded-full ${
                connectionStatus === "connected"
                  ? "bg-emerald-400"
                  : connectionStatus === "connecting"
                  ? "bg-amber-400"
                  : "bg-rose-400"
              }`}
            />
            <span className="capitalize">{connectionStatus}</span>
          </span>
          <span>• Whisper {whisperLoaded ? "ready" : "loading"}</span>
        </div>
      </div>
    </motion.div>
  );
}
