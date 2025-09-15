🧠 Cue – Cluely for Neuro-Spicy Devs

Cue is a neurodivergent-friendly AI assistant built for high-pressure moments — interviews, coding tests, and team meetings.
It provides real-time, stress-aware cognitive scaffolding through a minimal floating overlay, helping you stay focused, reduce overwhelm, and perform at your best without sensory overload.

✨ Core Value Proposition

"Your invisible thinking partner that reduces cognitive load during high-stress moments."

Cue is designed for neuro-spicy developers (ADHD, Autism, Anxiety) who want discreet support during cognitively demanding tasks — without compromising privacy, autonomy, or professional appearance.

🧩 Key Features
🧠 Cognitive Load Management

✅ Task Chunking – Breaks complex problems into simple steps

✅ Focus Nudges – Gentle reminders to refocus when drifting

✅ Overwhelm Detection – Spots stress signals (pauses, typing speed)

✅ Executive Function Support – Helps with planning and prioritization

💬 Real-Time Coaching Modes

🎤 Interview Mode – Suggests discreet STAR/behavioral responses

🧪 Coding Test Mode – Offers debugging patterns & timeboxing prompts

🧑‍💻 Meeting Mode – Summarizes key points and nudges participation

🌈 Sensory-Friendly Design

✨ Minimal translucent overlay UI

🎛️ Adjustable opacity and triggers

🔇 Optional non-verbal cues (e.g. haptic feedback)

🛠 Technical Overview

Frontend: Tauri (React + TypeScript) – lightweight always-on-top overlay

Backend: Python (FastAPI + WebSocket) – Whisper STT, Tesseract OCR, LLM (Ollama/OpenRouter)

Persistence: Planned lightweight SQLite for session logging

Security: 100% local-first — no cloud uploads, no telemetry, all processing on device

🚀 Getting Started

# Clone

git clone https://github.com/your-username/cue.git && cd cue-new

# Install frontend + run Tauri

npm install
npm run tauri dev

# (Optional) run backend manually for debugging

cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py

# Ensure Ollama is running with a local model

ollama serve &
ollama run gemma3:1b

📈 Roadmap

MVP → Floating overlay, live STT, session toggling, interview coaching

Phase 2 → Stress detection (pauses, typing speed), adaptive coaching

Phase 3 → Personalization, calendar integration, plugin system

❤️ Why Cue?
Cue is Cluely for neuro-spicy devs: built for people who think differently, process differently, and thrive differently.
It doesn’t try to change how you think — it scaffolds your cognition so you can show up as your best self.

🧠 Built With

Tauri (React + TS) – Overlay UI

FastAPI – Backend

Faster-Whisper – Real-time STT

Tesseract – OCR

Ollama / OpenRouter – LLMs

SQLite – Local persistence (planned)

📄 License
MIT — use freely, adapt compassionately.

💡 “Neurodivergence isn’t a flaw — it’s a different way of navigating the world. Cue exists to make that journey smoother, smarter, and less lonely.”
