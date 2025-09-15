import { useState } from "react";
import { motion } from "framer-motion";
import { Pause, Mic, Brain, Eye, EyeOff } from "lucide-react"; // icons

export default function CueBar() {
  const [isRecording, setIsRecording] = useState(false);
  const [showPanel, setShowPanel] = useState(true);

  return (
    <motion.div
      drag
      dragMomentum={false}
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="fixed top-6 left-1/2 -translate-x-1/2 w-[600px] rounded-full bg-black/40 backdrop-blur-md shadow-lg border border-white/20 p-2 flex items-center space-x-4 text-white cursor-move"
    >
      {/* Record / Pause */}
      <button
        className="p-2 hover:bg-white/10 rounded-full"
        onClick={() => setIsRecording(!isRecording)}
      >
        {isRecording ? <Pause size={18} /> : <Mic size={18} />}
      </button>

      {/* Timer */}
      <span className="text-sm font-mono">
        {isRecording ? "00:52" : "00:00"}
      </span>

      {/* Ask AI */}
      <button className="px-3 py-1 rounded-lg bg-white/20 hover:bg-white/30 text-sm flex items-center space-x-1">
        <Brain size={16} />
        <span>Ask AI</span>
      </button>

      {/* Show/Hide Toggle */}
      <button
        className="ml-auto px-3 py-1 rounded-lg bg-white/10 hover:bg-white/20 text-sm flex items-center space-x-1"
        onClick={() => setShowPanel(!showPanel)}
      >
        {showPanel ? <EyeOff size={16} /> : <Eye size={16} />}
        <span>{showPanel ? "Hide" : "Show"}</span>
      </button>
    </motion.div>
  );
}
