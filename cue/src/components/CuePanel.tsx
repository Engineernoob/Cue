import { motion, AnimatePresence } from "framer-motion";

interface CuePanelProps {
  show: boolean;
}

export default function CuePanel({ show }: CuePanelProps) {
  return (
    <AnimatePresence>
      {show && (
        <motion.div
          key="cue-panel"
          initial={{ opacity: 0, x: 60 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 60 }}
          transition={{ duration: 0.25, ease: "easeInOut" }}
          className="fixed top-24 right-10 w-[380px] rounded-2xl bg-black/40 backdrop-blur-md shadow-xl border border-white/20 p-4 text-white"
        >
          <h2 className="font-semibold flex items-center space-x-2 mb-2">
            <span>✨</span>
            <span>Live insights</span>
          </h2>

          <div className="text-sm space-y-3">
            <div>
              <h3 className="font-medium">API rollout timeline</h3>
              <ul className="list-disc list-inside text-white/80">
                <li>Neel gave an update on QA blockers</li>
                <li>You discussed next steps for rollout</li>
              </ul>
            </div>

            <div>
              <h3 className="font-medium">Actions</h3>
              <ul className="list-disc list-inside text-white/80">
                <li>📘 Define E2E testing</li>
                <li>❓ Fix invalid token error in QA</li>
                <li>💬 Suggest follow-up questions</li>
                <li>✨ What should I say next?</li>
              </ul>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
