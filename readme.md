# 🧠 Cue – Your Invisible Thinking Partner

**Cue** is a neurodivergent-friendly AI assistant designed for high-pressure situations like interviews, coding tests, and team meetings. It provides real-time, stress-aware cognitive scaffolding through a minimal floating interface — helping users regulate attention, reduce overwhelm, and perform at their best without sensory overload.

---

## ✨ Core Value Proposition

> _"Your invisible thinking partner that reduces cognitive load during high-stress moments."_

Cue is designed for neurodivergent users who want support during cognitively demanding tasks without compromising privacy, autonomy, or professional appearance.

---

## 🧩 Neurodivergent-Specific Features

### 🧠 Cognitive Load Management
- ✅ **Task Chunking** – Breaks complex tasks into simple steps
- ✅ **Focus Nudges** – Gentle refocus reminders based on user behavior
- ✅ **Overwhelm Detection** – Detects stress (pauses, typing speed) and suggests breaks
- ✅ **Executive Function Support** – Helps with planning, prioritization, and tracking

### 💬 Real-Time Coaching Modes
- 🎤 **Interview Mode** – Discreetly suggests behavioral/technical responses
- 🧪 **Coding Test Mode** – Offers debugging patterns, timeboxing prompts
- 🧑‍💻 **Meeting Mode** – Tracks participation, summarizes key points

### 🌈 Sensory-Friendly Design
- ✨ Minimalist translucent overlay UI
- 🎛️ Adjustable opacity and interaction triggers
- 🔇 Optional non-verbal feedback (e.g., haptics)

---

## 🧱 Architecture Overview
Audio/Screen Capture
↓
Context Recognition (Interview, Coding, Meeting)
↓
Neurodivergent-Aware Coaching Engine
↓
Discreet Visual/Haptic Prompts

---
### 🧩 Modular Plugin System
- `interview-plugin` — Suggests clarifications, STAR responses
- `coding-plugin` — Frameworks, edge case reminders
- `adhd-plugin` — Focus timers, hyperfocus exit prompts
- `autism-plugin` — Social cue interpretation, speaking scripts
- `anxiety-plugin` — Grounding techniques, confidence boosts

---

## 🛠 Technical Overview

- **Frontend**: Electron-based floating translucent bar with keyboard and mouse shortcuts
- **Backend**: Local Python WebSocket server with support for OpenRouter, Ollama, or local LLMs
- **Persistence**: Lightweight SQLite logging with session-based activity tracking
- **Security**: Fully local-first, with optional real-time feedback only during consented sessions

---

## 🚀 Getting Started

```bash
# Clone
git clone https://github.com/your-username/cue.git && cd cue

# Electron frontend
cd electron-frontend
npm install && npm start

# Python backend
cd ../python-backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python server.py

🔐 Privacy & Trust

Cue is designed for safety and dignity:
	•	🔒 Local-only: No cloud uploads, no third-party telemetry
	•	🙋 Explicit session control: User-initiated start/stop
	•	👁️ Transparent actions: Visual indicators show when capture is active
	•	🧼 Minimal data retention: Only coaching metadata is stored, never raw input

⸻

📈 Roadmap
Phase
Features
MVP
Floating UI, session toggling, interview coaching
Phase 2
Stress detection, adaptive coaching, typing/voice analysis
Phase 3
Learning user styles, calendar integration, plugin store

```
---
❤️ Why Cue?

Unlike generic productivity tools, Cue supports the way you think, not how you’re expected to think. It’s built for people with ADHD, Autism, Anxiety, or anyone who struggles with cognitive load under pressure — with an emphasis on privacy, agency, and trust.

⸻

🧠 Built With
	•	Electron – Floating UI bar
	•	WebSocket – Real-time backend communication
	•	FastAPI or Starlette – Python backend
	•	robotjs, desktopCapturer – Screen/audio analysis
	•	better-sqlite3 – Fast embedded local logging

⸻

🧑‍🎓 Inspired By
	•	Glass – Invisible UI pattern
	•	Cluely – Real-time attention intelligence
	•	Neurodivergent communities advocating for accessible tech

⸻
📄 License

MIT — use freely, adapt compassionately.

---

“Neurodivergence isn’t a flaw — it’s a different way of navigating the world. Cue exists to make that journey smoother, smarter, and less lonely.”