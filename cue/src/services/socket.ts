let socket: WebSocket | null = null;

export function initSocket(
  onMessage: (msg: any) => void,
  onError?: (err: any) => void
) {
  socket = new WebSocket("ws://127.0.0.1:8001/ws");

  socket.onopen = () => console.log("✅ Connected to Cue backend");
  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (e) {
      console.error("Invalid JSON from backend", e);
    }
  };
  socket.onerror = (err) => {
    console.error("Socket error:", err);
    onError?.(err);
  };
  socket.onclose = () => console.log("❌ Disconnected from backend");
}

export function sendMessage(msg: object) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(msg));
  } else {
    console.warn("⚠️ Socket not connected");
  }
}

export function closeSocket() {
  if (socket) {
    socket.close();
  }
}
