import { useState } from "react";
import { motion } from "framer-motion";

export default function Overlay() {
  const [query, setQuery] = useState("");

  return (
    <motion.div
      drag
      dragMomentum={false}
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="fixed top-10 left-1/2 -translate-x-1/2 w-[700px] rounded-2xl bg-white/20 backdrop-blur-md shadow-lg border border-white/30 p-3 flex items-center space-x-3 cursor-move"
    >
      {/* Input */}
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask Cue anything..."
        className="flex-1 bg-transparent text-white placeholder-white/60 outline-none px-2 text-lg"
      />

      {/* Button */}
      <button
        className={`px-4 py-1 rounded-xl text-sm font-medium transition ${
          query
            ? "bg-white/30 text-white hover:bg-white/40"
            : "bg-white/10 text-white/50 cursor-not-allowed"
        }`}
        disabled={!query}
        onClick={() => {
          console.log("Send:", query);
          setQuery("");
        }}
      >
        Ask
      </button>
    </motion.div>
  );
}
