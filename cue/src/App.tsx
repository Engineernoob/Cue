import { useState } from "react";
import CueBar from "../src/components/Cuebar";
import CuePanel from "./components/CuePanel";

function App() {
  const [showPanel, setShowPanel] = useState(true);

  return (
    <div className="w-screen h-screen">
      <CueBar showPanel={showPanel} setShowPanel={setShowPanel} />
      <CuePanel show={showPanel} />
    </div>
  );
}

export default App;
