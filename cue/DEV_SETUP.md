🔧 Developer Setup — Cue

This document is for contributors, collaborators, and neuro-spicy builders who want to run or extend Cue locally.

🖥️ Requirements
General

macOS or Linux (Windows support in progress)

Node.js >=18

Rust (via rustup
)

Python 3.10+

Python Dependencies

fastapi, uvicorn

faster-whisper

pillow (PIL)

numpy

pytesseract

webrtcvad

requests

Install system-level Tesseract OCR (macOS example):

brew install tesseract

LLM Runtime

Ollama
or OpenRouter

Example model: gemma3:1b

🏗️ Project Structure
Cue-New/
├── backend/ # Python backend (FastAPI + Whisper + OCR + LLM)
│ ├── main.py
│ └── requirements.txt
├── src/ # React + TypeScript frontend (Tauri webview)
│ ├── components/
│ ├── hooks/
│ ├── App.tsx
│ └── main.tsx
├── src-tauri/ # Rust shell (Tauri)
│ ├── src/
│ │ └── main.rs # Spawns Python backend + configures overlay window
│ └── tauri.conf.json
└── package.json

⚙️ Setup Instructions

1. Clone and Enter Project
   git clone https://github.com/your-username/cue.git
   cd cue-new

2. Setup Frontend (Tauri + React)
   npm install

Run Tauri dev mode:

npm run tauri dev

This:

Spawns the Python backend (backend/main.py)

Launches the Tauri overlay

3. Setup Backend (Python)

Create venv:

cd backend
python3 -m venv .venv
source .venv/bin/activate

Install requirements:

pip install -r requirements.txt

Run backend manually (optional debug mode):

uvicorn main:app --reload --host 127.0.0.1 --port 8001

Health check:

curl http://127.0.0.1:8001/health

4. Setup LLM (Ollama Example)

Install Ollama
then run:

ollama serve &
ollama run gemma3:1b

This ensures the model is downloaded and ready.

🧪 Testing the Connection

Start the app:

npm run tauri dev

Watch logs for:

Backend: Starting FastAPI backend...

Frontend: Connected to Cue backend ✅

Open overlay → click Ask AI → should stream LLM response from Ollama.

🛠️ Development Tips

Backend only: python backend/main.py

Frontend only: npm run dev

Tauri packaging: npm run tauri build (produces .app bundle on macOS)

Logging: Remove stdout(Stdio::null()) in main.rs if you want backend logs visible.

📈 Roadmap for Devs

Add SQLite for local session logging

Add hotkey toggle (⌘+Shift+C)

Add “plugins” (interview, coding, ADHD mode, etc.)

Package into signed .dmg for distribution

💡 Cue is neuro-spicy friendly — keep your PRs the same way: structured, supportive, and easy to follow.
