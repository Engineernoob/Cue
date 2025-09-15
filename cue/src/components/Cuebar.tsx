import { useState } from "react";
import { motion } from "framer-motion";
import { Pause, Mic } from "lucide-react";
import { sendMessage } from "../services/socket";
import RecordRTC from "recordrtc";

let recorder: any = null;

export default function CueBar() {
  const [isRecording, setIsRecording] = useState(false);

  async function toggleRecording() {
    if (!isRecording) {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recorder = new RecordRTC(stream, {
        type: "audio",
        mimeType: "audio/wav",
        recorderType: RecordRTC.StereoAudioRecorder,
        timeSlice: 1000, // every 1s
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
      recorder.startRecording();
    } else {
      recorder?.stopRecording();
    }
    setIsRecording(!isRecording);
  }

  return (
    <motion.div
      drag
      dragMomentum={false}
      className="fixed top-6 left-1/2 -translate-x-1/2 w-[600px] rounded-full bg-black/40 backdrop-blur-md shadow-lg border border-white/20 p-2 flex items-center space-x-4 text-white cursor-move"
    >
      <button
        className="p-2 hover:bg-white/10 rounded-full"
        onClick={toggleRecording}
      >
        {isRecording ? <Pause size={18} /> : <Mic size={18} />}
      </button>
      <span className="text-sm font-mono">
        {isRecording ? "Listening..." : "Idle"}
      </span>
    </motion.div>
  );
}
