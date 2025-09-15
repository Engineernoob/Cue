import { useEffect, useState } from "react";
import CueBar from "./components/Cuebar";
import CuePanel from "./components/CuePanel";
import {
  connect,
  onMessage,
  sendQuery,
  startSession,
  stopSession,
} from "../src/services/socket";

export type CueMessage = {
  type: "transcript" | "ai" | "auto_hint";
  text: string;
  streaming?: boolean;
};

function App() {
  const [showPanel, setShowPanel] = useState(true);
  const [messages, setMessages] = useState<CueMessage[]>([]);

  useEffect(() => {
    connect();

    const unsubscribe = onMessage((msg) => {
      switch (msg.type) {
        case "transcript":
          setMessages((prev) => [
            ...prev,
            { type: "transcript", text: msg.text },
          ]);
          break;

        case "llm_response_chunk":
          setMessages((prev) => [
            ...prev,
            { type: "ai", text: msg.text_chunk, streaming: true },
          ]);
          break;

        case "llm_response_complete":
          setMessages((prev) => [
            ...prev.filter((m) => !(m.type === "ai" && m.streaming)),
            { type: "ai", text: msg.text, streaming: false },
          ]);
          break;

        case "auto_hint_chunk":
          setMessages((prev) => [
            ...prev,
            { type: "auto_hint", text: msg.text_chunk, streaming: true },
          ]);
          break;

        case "auto_hint_complete":
          setMessages((prev) => [
            ...prev.filter((m) => !(m.type === "auto_hint" && m.streaming)),
            { type: "auto_hint", text: msg.text, streaming: false },
          ]);
          break;

        default:
          console.warn("⚠️ Unhandled message:", msg);
      }
    });

    return () => unsubscribe();
  }, []);

  return (
    <div className="w-screen h-screen">
      <CueBar
        setShowPanel={setShowPanel}
        onStart={startSession}
        onStop={stopSession}
        onSendQuery={sendQuery}
      />
      <CuePanel show={showPanel} messages={messages} />
    </div>
  );
}

export default App;
