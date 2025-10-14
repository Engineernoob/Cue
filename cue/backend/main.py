"""FastAPI backend for Cue."""

from __future__ import annotations

import asyncio
import base64
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import io
import json
import logging
import os
import re
import threading
import time
import uuid
import wave
from typing import Any, AsyncIterator, Deque

import numpy as np
from numpy.typing import NDArray
import requests
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from PIL import Image
from faster_whisper import WhisperModel
from pytesseract import pytesseract  # pyright: ignore[reportMissingTypeStubs]


# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --- Config ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "50"))
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

ALLOWED_ORIGINS = {
    origin.strip()
    for origin in os.getenv("CUE_ALLOWED_ORIGINS", "http://localhost:1420, tauri://localhost").split(",")
    if origin.strip()
}

CONTEXT_BUFFER_MAX_LENGTH = 2000
CONTEXT_BUFFER_MAX_TIME_SECONDS = 300
OCR_MAX_CHARACTERS = 2000

SAMPLE_RATE = 16000
MIN_BUFFER_LENGTH = float(os.getenv("CUE_MIN_AUDIO_SECONDS", "0.75"))
VAD_FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)
FRAME_BYTES = FRAME_SAMPLES * 2
HEARTBEAT_INTERVAL_SECONDS = 15

RATE_LIMIT_MAX_REQUESTS = int(os.getenv("CUE_RATE_LIMIT", "15"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("CUE_RATE_LIMIT_WINDOW", "60"))


# --- Utilities ---
def sanitize_text(text: str, limit: int = OCR_MAX_CHARACTERS) -> str:
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > limit:
        cleaned = f"{cleaned[:limit]}…"
    return cleaned


def detect_context_type(context: str) -> str:
    context_lower = context.lower()
    platforms = ["leetcode", "hackerrank", "codility", "codesignal", "hackerearth", "geeksforgeeks"]
    if any(p in context_lower for p in platforms):
        return "coding_assessment"
    if any(k in context_lower for k in ["error", "bug", "debug", "exception"]):
        return "debugging"
    if any(k in context_lower for k in ["two pointers", "sliding window", "binary search", "backtracking"]):
        return "algorithm_pattern"
    if any(k in context_lower for k in ["algorithm", "function", "array", "tree", "graph"]):
        return "coding"
    if any(k in context_lower for k in ["interview", "tell me about", "experience"]):
        return "interview"
    return "general"


def build_neurodivergent_prompt(context: str, query: str, context_type: str, style: str | None = None) -> str:
    base_instructions = (
        "You are Cue, an AI assistant for neurodivergent developers."
        "\n- Provide clear, structured answers"
        "\n- Use step-by-step explanations"
        "\n- Reduce anxiety with supportive tone"
        "\n- Offer focus & executive function help"
    )

    style_instructions = ""
    if style:
        style_upper = style.upper()
        if style_upper == "STAR":
            style_instructions = "\nUse STAR format: Situation, Task, Action, Result."
        elif style_upper == "PAR":
            style_instructions = "\nUse PAR format: Problem, Action, Result."
        elif style_upper == "SCQA":
            style_instructions = "\nUse SCQA: Situation, Complication, Question, Answer."

    return (
        f"{base_instructions}{style_instructions}\n\n"
        f"Context type: {context_type}\n"
        f"Context: {context}\n\n"
        f"User query: \"{query}\"\n\n"
        "Respond concisely, supportively, and clearly."
    )


def is_speech(audio: NDArray[np.float32], vad: webrtcvad.Vad) -> bool:
    try:
        if len(audio) == 0:
            return False
        int16_bytes = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        if len(int16_bytes) < FRAME_BYTES:
            return False
        for offset in range(0, len(int16_bytes) - (len(int16_bytes) % FRAME_BYTES), FRAME_BYTES):
            frame = int16_bytes[offset : offset + FRAME_BYTES]
            if vad.is_speech(frame, SAMPLE_RATE):  # pyright: ignore[reportUnknownMemberType]
                return True
        return False
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.error("Error checking speech: %s", exc)
        return False


# --- State Containers ---
class AudioBuffer:
    def __init__(self, min_duration: float) -> None:
        self._chunks: list[NDArray[np.float32]] = []
        self._lock = asyncio.Lock()
        self._min_duration = min_duration

    async def append(self, chunk: NDArray[np.float32]) -> float:
        async with self._lock:
            self._chunks.append(chunk)
            total_samples = sum(len(buf) for buf in self._chunks)
        return total_samples / SAMPLE_RATE

    async def flush(self) -> NDArray[np.float32] | None:
        async with self._lock:
            if not self._chunks:
                return None
            combined = np.concatenate(self._chunks)
            self._chunks.clear()
        return combined

    @property
    def min_duration(self) -> float:
        return self._min_duration


class ContextStore:
    def __init__(self, max_length: int, max_age_seconds: int) -> None:
        self._buffer: Deque[tuple[float, str, str]] = deque()
        self._lock = asyncio.Lock()
        self._max_length = max_length
        self._max_age = max_age_seconds

    async def add(self, text_type: str, text: str) -> None:
        ts = time.time()
        async with self._lock:
            self._buffer.append((ts, text_type, text))
            current_length = sum(len(item[2]) for item in self._buffer)
            while current_length > self._max_length and len(self._buffer) > 1:
                removed = self._buffer.popleft()
                current_length -= len(removed[2])
            cutoff = time.time() - self._max_age
            while self._buffer and self._buffer[0][0] < cutoff:
                self._buffer.popleft()

    async def snapshot(self) -> str:
        async with self._lock:
            return "\n".join(item[2] for item in self._buffer)


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self._max_requests = max_requests
        self._window = window_seconds
        self._timestamps: Deque[float] = deque()
        self._lock = asyncio.Lock()

    async def allow(self) -> bool:
        now = time.monotonic()
        async with self._lock:
            while self._timestamps and now - self._timestamps[0] > self._window:
                self._timestamps.popleft()
            if len(self._timestamps) >= self._max_requests:
                return False
            self._timestamps.append(now)
        return True


class SessionState:
    def __init__(self) -> None:
        self._active = False
        self._lock = asyncio.Lock()

    async def set_active(self, active: bool) -> None:
        async with self._lock:
            self._active = active

    async def is_active(self) -> bool:
        async with self._lock:
            return self._active


@dataclass
class AppResources:
    context: ContextStore
    audio: AudioBuffer
    vad: webrtcvad.Vad
    rate_limiter: RateLimiter
    session: SessionState
    whisper_model: WhisperModel | None = None
    whisper_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def ensure_whisper(self) -> WhisperModel:
        if self.whisper_model:
            return self.whisper_model
        async with self.whisper_lock:
            if self.whisper_model:
                return self.whisper_model
            try:
                model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
                self.whisper_model = model
                logging.info("Whisper model loaded.")
                return model
            except Exception as exc:  # pragma: no cover - external dependency
                logging.error("Failed to load Whisper: %s", exc)
                raise


resources: AppResources | None = None


def get_resources() -> AppResources:
    if resources is None:
        raise RuntimeError("Application resources not initialised")
    return resources


# --- External Calls ---
def _ollama_worker(prompt: str, request_id: str, queue: asyncio.Queue[Any], loop: asyncio.AbstractEventLoop) -> None:
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            headers={"Content-Type": "application/json"},
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
            stream=True,
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        for chunk in response.iter_lines():
            if not chunk:
                continue
            try:
                payload = json.loads(chunk.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            loop.call_soon_threadsafe(queue.put_nowait, {"request_id": request_id, "payload": payload})
        loop.call_soon_threadsafe(queue.put_nowait, {"request_id": request_id, "payload": {"done": True}})
    except Exception as exc:
        loop.call_soon_threadsafe(queue.put_nowait, {"request_id": request_id, "error": str(exc)})


async def stream_ollama(prompt: str, request_id: str) -> AsyncIterator[dict[str, Any]]:
    queue: asyncio.Queue[Any] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    thread = threading.Thread(target=_ollama_worker, args=(prompt, request_id, queue, loop), daemon=True)
    thread.start()
    while True:
        message = await queue.get()
        if "error" in message:
            raise RuntimeError(message["error"])
        payload = message["payload"]
        yield payload
        if payload.get("done"):
            break


# --- FastAPI App ---


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # pragma: no cover - startup hook
    global resources
    resources = AppResources(
        context=ContextStore(CONTEXT_BUFFER_MAX_LENGTH, CONTEXT_BUFFER_MAX_TIME_SECONDS),
        audio=AudioBuffer(MIN_BUFFER_LENGTH),
        vad=webrtcvad.Vad(2),
        rate_limiter=RateLimiter(RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SECONDS),
        session=SessionState(),
    )
    app.state.resources = resources

    loop = asyncio.get_running_loop()

    async def _load_whisper() -> None:
        try:
            await resources.ensure_whisper()
        except Exception as exc:
            logging.error("Background whisper load failed: %s", exc)

    loop.create_task(_load_whisper())
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health_check() -> dict[str, Any]:
    res = get_resources()
    return {
        "status": "ok",
        "whisper_loaded": res.whisper_model is not None,
        "session_active": await res.session.is_active(),
    }


async def heartbeat(websocket: WebSocket) -> None:
    try:
        res = get_resources()
        while True:
            await websocket.send_json(
                {
                    "type": "heartbeat",
                    "timestamp": time.time(),
                    "whisper_loaded": res.whisper_model is not None,
                    "session_active": await res.session.is_active(),
                }
            )
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
    except Exception:
        return


def _validate_origin(websocket: WebSocket) -> bool:
    origin = websocket.headers.get("origin")
    return origin is None or origin in ALLOWED_ORIGINS


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    res = get_resources()

    if not _validate_origin(websocket):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        logging.warning("Rejected websocket from origin %s", websocket.headers.get("origin"))
        return

    await websocket.accept()
    logging.info("WebSocket connected.")
    heartbeat_task = asyncio.create_task(heartbeat(websocket))
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "audio_chunk":
                await handle_audio_chunk(websocket, message)
            elif msg_type == "image_data":
                await handle_image_data(websocket, message)
            elif msg_type == "llm_query":
                await handle_llm_query(websocket, message)
            elif msg_type == "start_session":
                await res.session.set_active(True)
                await websocket.send_json({"type": "session_status", "status": "started"})
            elif msg_type == "stop_session":
                await res.session.set_active(False)
                await websocket.send_json({"type": "session_status", "status": "stopped"})
            else:
                await websocket.send_json({"type": "error", "message": f"Unknown type: {msg_type}"})
    except WebSocketDisconnect:
        logging.info("WebSocket disconnected.")
    finally:
        heartbeat_task.cancel()


# --- Handlers ---


async def handle_audio_chunk(websocket: WebSocket, message: dict[str, Any]) -> None:
    try:
        res = get_resources()
        model = await res.ensure_whisper()
    except Exception as exc:
        await websocket.send_json({"type": "transcript_error", "message": f"Whisper unavailable: {exc}"})
        return

    try:
        data = message.get("data")
        if not isinstance(data, str):
            raise ValueError("Missing audio data")
        audio_bytes = base64.b64decode(data, validate=True)
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            pcm16: NDArray[np.int16] = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        audio_np: NDArray[np.float32] = pcm16.astype(np.float32) / 32768.0

        if not is_speech(audio_np, res.vad):
            return

        total_duration = await res.audio.append(audio_np)
        if total_duration < res.audio.min_duration:
            return

        combined = await res.audio.flush()
        if combined is None or len(combined) == 0:
            return

        segments, _info = model.transcribe(combined, beam_size=5)
        transcript = "".join(segment.text for segment in segments)
        transcript = sanitize_text(transcript)
        if not transcript:
            return

        await res.context.add("audio", transcript)
        message_id = str(uuid.uuid4())
        await websocket.send_json({"type": "transcript", "text": transcript, "message_id": message_id})
        if await res.session.is_active():
            asyncio.create_task(auto_llm_suggestion(websocket, transcript))
    except Exception as exc:
        logging.error("Audio error: %s", exc, exc_info=True)
        await websocket.send_json({"type": "transcript_error", "message": str(exc)})


async def handle_image_data(websocket: WebSocket, message: dict[str, Any]) -> None:
    try:
        res = get_resources()
        data = message.get("data")
        if not isinstance(data, str):
            raise ValueError("Missing image data")
        raw = base64.b64decode(data)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        extracted = pytesseract.image_to_string(image)  # pyright: ignore[reportUnknownMemberType]
        cleaned = sanitize_text(extracted)
        if cleaned:
            await res.context.add("screen", cleaned)
            payload = {"type": "ocr_result", "text": cleaned, "message_id": str(uuid.uuid4())}
            await websocket.send_json(payload)
            if await res.session.is_active():
                asyncio.create_task(auto_llm_suggestion(websocket, cleaned))
    except Exception as exc:
        logging.error("OCR error: %s", exc, exc_info=True)
        await websocket.send_json({"type": "ocr_error", "message": str(exc)})


async def handle_llm_query(websocket: WebSocket, message: dict[str, Any]) -> None:
    res = get_resources()

    if not await res.rate_limiter.allow():
        await websocket.send_json({"type": "llm_response_error", "message": "Rate limit exceeded. Please wait."})
        return

    query = message.get("query", "")
    if not isinstance(query, str) or not query.strip():
        await websocket.send_json({"type": "llm_response_error", "message": "Empty query."})
        return

    style = message.get("style")
    context = await res.context.snapshot()
    prompt = build_neurodivergent_prompt(context, query, detect_context_type(context), style)
    request_id = str(uuid.uuid4())
    attempts = 0
    accumulated: list[str] = []

    while attempts <= OLLAMA_MAX_RETRIES:
        try:
            async for payload in stream_ollama(prompt, request_id):
                token = payload.get("response", "")
                if token:
                    accumulated.append(token)
                    await websocket.send_json(
                        {
                            "type": "llm_response",
                            "status": "chunk",
                            "response_id": request_id,
                            "text": token,
                        }
                    )
                if payload.get("done"):
                    full = sanitize_text("".join(accumulated))
                    if full:
                        await res.context.add("llm_response", full)
                        await websocket.send_json(
                            {
                                "type": "llm_response",
                                "status": "complete",
                                "response_id": request_id,
                                "text": full,
                            }
                        )
                    return
        except Exception as exc:
            attempts += 1
            logging.warning("LLM attempt %s failed: %s", attempts, exc)
            if attempts > OLLAMA_MAX_RETRIES:
                await websocket.send_json({"type": "llm_response_error", "message": f"LLM error: {exc}"})
                return
            await asyncio.sleep(1.5 * attempts)


async def auto_llm_suggestion(websocket: WebSocket, latest_text: str) -> None:
    request_id = str(uuid.uuid4())
    res = get_resources()
    context = await res.context.snapshot()
    prompt = build_neurodivergent_prompt(context, f"Give a short hint: {latest_text}", detect_context_type(context))
    accumulated: list[str] = []
    try:
        async for payload in stream_ollama(prompt, request_id):
            token = payload.get("response", "")
            if token:
                accumulated.append(token)
                await websocket.send_json(
                    {
                        "type": "auto_hint",
                        "status": "chunk",
                        "hint_id": request_id,
                        "text": token,
                    }
                )
            if payload.get("done"):
                full = sanitize_text("".join(accumulated))
                if full:
                    await res.context.add("auto_hint", full)
                    await websocket.send_json(
                        {
                            "type": "auto_hint",
                            "status": "complete",
                            "hint_id": request_id,
                            "text": full,
                        }
                    )
                return
    except Exception as exc:
        logging.error("Auto-hint error: %s", exc)
        await websocket.send_json({"type": "auto_hint_error", "message": str(exc)})


if __name__ == "__main__":
    import uvicorn

    logging.info("Starting backend on %s with %s", OLLAMA_HOST, OLLAMA_MODEL)
    uvicorn.run(app, host="127.0.0.1", port=8001, ws="websockets")
