import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { CueMessage } from "../App";
import { sendQuery } from "../services/socket";

export default function CuePanel({
  show,
  messages,
}: {
  show: boolean;
  messages: CueMessage[];
}) {
  const panelRef = useRef<HTMLDivElement>(null);
  const [input, setInput] = useState("");

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (panelRef.current) {
      panelRef.current.scrollTop = panelRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      sendQuery(input);
      setInput("");
    }
  };

  // Check if AI or Hint is streaming
  const isStreaming = messages.some((m) => m.streaming);

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          key="cue-panel"
          initial={{ opacity: 0, x: 60 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 60 }}
          transition={{ duration: 0.25 }}
          className="fixed top-24 right-10 w-[400px] h-[500px] rounded-2xl bg-black/40 backdrop-blur-md shadow-xl border border-white/20 p-4 text-white flex flex-col"
        >
          <h2 className="font-semibold mb-3">🧠 Cue Panel</h2>

          {/* Messages */}
          <div
            ref={panelRef}
            className="flex-1 overflow-y-auto space-y-3 text-sm pr-2"
          >
            {messages.map((m, i) => (
              <div
                key={i}
                className={`p-2 rounded-md ${
                  m.type === "transcript"
                    ? "bg-white/10 text-white/80"
                    : m.type === "ai"
                    ? "bg-blue-500/20 text-blue-200"
                    : "bg-yellow-500/20 text-yellow-200 italic"
                }`}
              >
                {m.type === "auto_hint" && (
                  <span className="font-bold">💡 Hint: </span>
                )}
                {m.type === "ai" && <span className="font-bold">🤖 AI: </span>}
                {m.type === "transcript" && (
                  <span className="font-bold">🎤 You: </span>
                )}
                {m.text}
              </div>
            ))}

            {/* Loading Indicator */}
            {isStreaming && (
              <div className="flex items-center gap-1 text-blue-300 italic">
                <span>🤖 Cue is thinking</span>
                <span className="flex gap-1">
                  <span className="dot animate-bounce delay-0">.</span>
                  <span className="dot animate-bounce delay-150">.</span>
                  <span className="dot animate-bounce delay-300">.</span>
                </span>
              </div>
            )}
          </div>

          {/* Input Box */}
          <form onSubmit={handleSend} className="mt-3 flex gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask Cue..."
              className="flex-1 px-3 py-2 rounded-md bg-white/10 text-white placeholder-white/40 outline-none"
            />
            <button
              type="submit"
              className="px-4 py-2 bg-blue-500 hover:bg-blue-600 rounded-md text-white"
            >
              Send
            </button>
          </form>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
