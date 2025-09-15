import { useEffect, useState } from "react";
import { initSocket } from "../services/socket";
import { motion, AnimatePresence } from "framer-motion";

export default function CuePanel({ show }: { show: boolean }) {
  const [transcripts, setTranscripts] = useState<string[]>([]);

  useEffect(() => {
    initSocket((msg) => {
      if (msg.type === "transcript") {
        setTranscripts((prev) => [...prev, msg.text]);
      }
    });
  }, []);

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          key="cue-panel"
          initial={{ opacity: 0, x: 60 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 60 }}
          transition={{ duration: 0.25 }}
          className="fixed top-24 right-10 w-[380px] h-[500px] rounded-2xl bg-black/40 backdrop-blur-md shadow-xl border border-white/20 p-4 text-white overflow-y-auto"
        >
          <h2 className="font-semibold mb-3">✨ Live Transcript</h2>
          <div className="space-y-2 text-sm">
            {transcripts.map((t, i) => (
              <p key={i} className="text-white/80">
                {t}
              </p>
            ))}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
