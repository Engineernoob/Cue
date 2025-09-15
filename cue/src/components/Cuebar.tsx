import { useState, useRef, useEffect, Dispatch, SetStateAction } from "react";
import { motion } from "framer-motion";
import { Pause, Mic, Brain } from "lucide-react";
import { sendMessage } from "../services/socket";
import RecordRTC from "recordrtc";

interface Props {
  setShowPanel: Dispatch<SetStateAction<boolean>>;
  onStart: () => void;
  onStop: () => void;
  onSendQuery: (query: string) => void;
}

export default function CueBar({
  setShowPanel,
  onStart,
  onStop,
  onSendQuery,
}: Props) {
  const [isRecording, setIsRecording] = useState(false);
  const [query, setQuery] = useState("");
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<any>(null);

  useEffect(() => {
    return () => {
      if (recorderRef.current) {
        recorderRef.current.stopRecording();
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop());
          streamRef.current = null;
        }
        recorderRef.current = null;
      }
    };
  }, []);

  async function toggleRecording() {
    if (!isRecording) {
      try {
        onStart();
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: true,
        });
        streamRef.current = stream;
        recorderRef.current = new RecordRTC(stream, {
          type: "audio",
          mimeType: "audio/wav",
          recorderType: RecordRTC.StereoAudioRecorder,
          timeSlice: 1000,
          desiredSampRate: 16000,
          numberOfAudioChannels: 1,
          ondataavailable: (blob: Blob) => {
            blob.arrayBuffer().then((buf) => {
              const bytes = new Uint8Array(buf);
              const b64 = btoa(String.fromCharCode(...bytes));
              sendMessage({ type: "audio_chunk", data: b64 });
            });
          },
        });
        recorderRef.current.startRecording();
        setIsRecording(true);
      } catch (error) {
        console.error("Failed to access microphone:", error);
        onStop();
      }
    } else {
      recorderRef.current?.stopRecording();
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      recorderRef.current = null;
      onStop();
      setIsRecording(false);
    }
  }

  function askAI() {
    if (query.trim().length > 0) {
      onSendQuery(query);
      setQuery("");
      setShowPanel(true);
    }
  }

  return (
    <motion.div
      drag
      dragMomentum={false}
      className="fixed top-6 left-1/2 -translate-x-1/2 w-[650px] rounded-full bg-black/40 backdrop-blur-md shadow-lg border border-white/20 p-2 flex items-center space-x-4 text-white cursor-move"
    >
      {/* Record / Pause */}
      <button
        className="p-2 hover:bg-white/10 rounded-full"
        onClick={toggleRecording}
      >
        {isRecording ? <Pause size={18} /> : <Mic size={18} />}
      </button>
      <span className="text-sm font-mono">
        {isRecording ? "Listening..." : "Idle"}
      </span>

      {/* Ask AI */}
      <div className="flex items-center space-x-2 flex-1 ml-4">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              askAI();
            }
          }}
          placeholder="Ask Cue anything..."
          className="flex-1 px-3 py-1 rounded-lg bg-white/10 text-sm focus:outline-none"
        />
        <button
          className="px-3 py-1 rounded-lg bg-white/20 hover:bg-white/30 text-sm flex items-center space-x-1"
          onClick={askAI}
        >
          <Brain size={16} />
          <span>Ask AI</span>
        </button>
      </div>
    </motion.div>
  );
}
