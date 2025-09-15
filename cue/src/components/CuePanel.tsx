import { motion, AnimatePresence } from "framer-motion";
import { CueMessage } from "../App";

export default function CuePanel({
  show,
  messages,
}: {
  show: boolean;
  messages: CueMessage[];
}) {
  const transcripts = messages
    .filter((m) => m.type === "transcript")
    .map((m) => m.text);
  const aiMessages = messages.filter((m) => m.type === "ai");
  const latestAi = aiMessages[aiMessages.length - 1];
  const aiResponse = latestAi ? latestAi.text : "";

  const autoHints = messages.filter((m) => m.type === "auto_hint");
  const latestHint = autoHints[autoHints.length - 1];
  const hintText = latestHint ? latestHint.text : "";

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          key="cue-panel"
          initial={{ opacity: 0, x: 60 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 60 }}
          transition={{ duration: 0.25 }}
          className="fixed top-24 right-10 w-[400px] h-[500px] rounded-2xl bg-black/40 backdrop-blur-md shadow-xl border border-white/20 p-4 text-white overflow-y-auto"
        >
          <h2 className="font-semibold mb-3">✨ Live Transcript</h2>
          <div className="space-y-2 text-sm mb-4">
            {transcripts.map((t, i) => (
              <p key={i} className="text-white/80">
                {t}
              </p>
            ))}
          </div>

          <h2 className="font-semibold mb-3">🤖 Cue AI</h2>
          <div className="text-sm whitespace-pre-wrap mb-4">
            {aiResponse || "Ask Cue a question using the bar above."}
          </div>

          {hintText && (
            <>
              <h2 className="font-semibold mb-3">💡 Auto Hint</h2>
              <div className="text-sm whitespace-pre-wrap text-blue-300">
                {hintText}
              </div>
            </>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
