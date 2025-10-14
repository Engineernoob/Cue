import { motion, AnimatePresence } from "framer-motion";
import {
  useEffect,
  useRef,
  useState,
  Dispatch,
  SetStateAction,
  FormEvent,
} from "react";
import { CueMessage } from "../App";
import { sendQuery } from "../services/socket";

interface CuePanelProps {
  show: boolean;
  messages: CueMessage[];
  hintsEnabled: boolean;
  onToggleHints: Dispatch<SetStateAction<boolean>>;
  heartbeatLabel: string;
}

const EMPTY_STATE = "No messages yet. Start speaking or ask a question to begin.";

export default function CuePanel({
  show,
  messages,
  hintsEnabled,
  onToggleHints,
  heartbeatLabel,
}: CuePanelProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const [input, setInput] = useState("");

  useEffect(() => {
    if (panelRef.current) {
      panelRef.current.scrollTop = panelRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!input.trim()) return;
    sendQuery(input.trim());
    setInput("");
  };

  const streaming = messages.some((message) => message.streaming);

  const messageStyles: Record<CueMessage["type"], string> = {
    transcript:
      "bg-gradient-to-r from-slate-900/70 via-slate-800/50 to-slate-900/60 text-slate-100",
    screen:
      "bg-gradient-to-r from-indigo-500/25 via-purple-500/20 to-slate-900/55 text-indigo-100",
    ai: "bg-gradient-to-r from-blue-500/25 via-sky-500/20 to-slate-900/55 text-blue-100",
    auto_hint:
      "bg-gradient-to-r from-amber-400/20 via-amber-300/15 to-amber-500/25 text-amber-100 italic",
    system: "bg-white/10 text-slate-200",
  };

  const messageIcons: Record<CueMessage["type"], string> = {
    transcript: "🎤",
    screen: "🖼️",
    ai: "🤖",
    auto_hint: "💡",
    system: "⚙️",
  };

  return (
    <AnimatePresence>
      {show && (
        <motion.section
          key="cue-panel"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 24 }}
          transition={{ duration: 0.26, ease: "easeOut" }}
          className="flex h-full flex-col rounded-[32px] border border-white/10 bg-white/[0.08] p-6 text-white shadow-[0_40px_120px_rgba(10,21,55,0.45)] backdrop-blur-2xl"
          aria-label="Cue conversation panel"
        >
          <header className="flex items-center justify-between gap-3">
            <div className="flex flex-col">
              <h2 className="text-xl font-semibold tracking-wide">Live insights</h2>
              <p className="text-xs text-slate-200/80" aria-live="polite">
                {heartbeatLabel}
              </p>
            </div>
            <button
              type="button"
              onClick={() => onToggleHints((prev) => !prev)}
              className="group flex items-center gap-2 rounded-full border border-amber-200/40 bg-amber-400/15 px-4 py-2 text-xs font-medium uppercase tracking-wide text-amber-100 transition-all duration-200 hover:border-amber-200/70 hover:bg-amber-400/25 hover:shadow-[0_0_18px_rgba(251,191,36,0.35)]"
              aria-pressed={hintsEnabled}
              aria-label={hintsEnabled ? "Disable auto hints" : "Enable auto hints"}
            >
              <span className="text-base leading-none">💡</span>
              {hintsEnabled ? "Hints on" : "Hints off"}
            </button>
          </header>

          <div
            ref={panelRef}
            className="mt-5 flex-1 space-y-3 overflow-y-auto pr-2 text-sm"
            role="log"
            aria-live="polite"
          >
            {messages.length === 0 && (
              <motion.p
                initial={{ opacity: 0.6, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, ease: "easeOut" }}
                className="rounded-2xl border border-white/5 bg-white/5 px-6 py-8 text-center text-slate-200/80 shadow-inner"
              >
                {EMPTY_STATE}
              </motion.p>
            )}

            {messages.map((message) => (
              <motion.article
                key={message.id}
                layout
                initial={{ opacity: 0, y: 14, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.22, ease: "easeOut" }}
                className={`group relative overflow-hidden rounded-2xl border border-white/10 px-4 py-3 leading-relaxed shadow-[0_12px_40px_rgba(8,18,46,0.35)] backdrop-blur ${messageStyles[message.type]}`}
              >
                <div className="absolute inset-0 bg-white/4 opacity-0 transition-opacity duration-200 group-hover:opacity-100" />
                <div className="relative flex items-start gap-3">
                  <span className="mt-1 text-lg">{messageIcons[message.type]}</span>
                  <div className="flex flex-col gap-1">
                    <span>{message.text}</span>
                    {message.streaming && (
                      <span className="flex items-center gap-1 text-xs text-blue-100/80">
                        <span className="inline-flex h-1.5 w-1.5 rounded-full bg-blue-200 animate-pulse" />
                        streaming…
                      </span>
                    )}
                  </div>
                </div>
              </motion.article>
            ))}

            {streaming && (
              <div className="flex items-center gap-2 text-blue-200/80">
                <span className="text-sm uppercase tracking-wide">Cue is responding</span>
                <span className="flex items-center gap-1">
                  <span className="h-1.5 w-1.5 rounded-full bg-blue-200 animate-bounce" />
                  <span className="h-1.5 w-1.5 rounded-full bg-blue-200 animate-bounce [animation-delay:120ms]" />
                  <span className="h-1.5 w-1.5 rounded-full bg-blue-200 animate-bounce [animation-delay:240ms]" />
                </span>
              </div>
            )}
          </div>

          <form onSubmit={handleSend} className="mt-5 flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 backdrop-blur">
            <label htmlFor="cue-panel-input" className="sr-only">
              Ask Cue
            </label>
            <input
              id="cue-panel-input"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Ask Cue…"
              className="flex-1 rounded-xl bg-transparent px-2 text-sm outline-none transition placeholder:text-slate-300/60 focus:ring-0"
            />
            <button
              type="submit"
              className="rounded-full bg-gradient-to-r from-blue-500 via-indigo-500 to-sky-400 px-5 py-2 text-sm font-semibold text-white shadow-[0_12px_30px_rgba(15,76,177,0.45)] transition hover:shadow-[0_18px_40px_rgba(23,93,205,0.55)] disabled:cursor-not-allowed disabled:bg-white/20 disabled:text-slate-200/60 disabled:shadow-none"
              disabled={!input.trim()}
            >
              Send
            </button>
          </form>
        </motion.section>
      )}
    </AnimatePresence>
  );
}
