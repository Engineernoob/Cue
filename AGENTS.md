# Repository Guidelines

## Project Structure & Module Organization
- `electron-frontend/` — Electron app (entry `main.js`, `renderer.js`, `preload.js`, UI `index.html`, `style.css`, config via `config.js`, backend orchestration in `backendManager.js`).
- `backend/` — FastAPI + WebSocket server (`main.py`) and Python deps (`requirements.txt`).
- Root docs: `readme.md`, `SETUP.md`.

## Build, Test, and Development Commands
- Install + run (frontend starts backend automatically):
  - `cd electron-frontend && npm install && npm start`
- Backend only (from repo root):
  - `npm --prefix electron-frontend run backend-only`
  - Or manual: `python3 -m venv .venv && source .venv/bin/activate && pip install -r backend/requirements.txt && python backend/main.py`
- Package desktop app: `cd electron-frontend && npm run build` (artifacts in `electron-frontend/dist/`).
- Quick backend smoke test: `cd electron-frontend && npm run test-backend`.

## Coding Style & Naming Conventions
- JavaScript: 2-space indent; camelCase for files (`screenMonitor.js`), vars, and functions; PascalCase for classes.
- Python: 4-space indent; snake_case for modules/functions; PascalCase for classes; UPPER_SNAKE_CASE for constants.
- Keep functions focused; prefer clear logs (`logging` in backend) and short, descriptive names.

## Testing Guidelines
- No formal suite yet. When adding:
  - Python: place tests in `backend/tests/`, name `test_*.py` (pytest recommended).
  - Frontend: place tests in `electron-frontend/__tests__/`, name `*.test.js` (Jest recommended).
- Sanity checks: ensure `GET http://127.0.0.1:8001/health` returns `{ "status": "ok" }` and Electron connects to `ws://127.0.0.1:8001/ws`.

## Commit & Pull Request Guidelines
- Use Conventional Commits: `feat(frontend): ...`, `fix(backend): ...`, `chore: ...`, `docs: ...`.
- PRs must include: concise description, scope (frontend/backend), reproduction steps, screenshots (UI) or logs, linked issues, and any env/config notes.
- Keep diffs minimal and focused; update `readme.md`/`SETUP.md` when behavior changes.

## Security & Configuration Tips
- Backend env vars: `OLLAMA_HOST`, `OLLAMA_MODEL`, `WHISPER_MODEL_SIZE`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`.
- OCR requires Tesseract installed on the system (`pytesseract` uses system binary).
- Do not commit secrets or large model artifacts. Prefer local, default settings in `electron-frontend/config.js` (electron-store).

