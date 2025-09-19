# backend/main.py
import asyncio
import base64
from collections import deque
import io
import json
import logging
import os
import threading
import time
import wave
from typing import Any

from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import numpy as np
from numpy.typing import NDArray
from pytesseract import pytesseract  # pyright: ignore[reportMissingTypeStubs]
import requests
import webrtcvad
from contextlib import asynccontextmanager

#--- Logging---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#--- Config ---
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma3:1b")

WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

CONTEXT_BUFFER_MAX_LENGTH = 2000
CONTEXT_BUFFER_MAX_TIME_SECONDS = 300

# --- State ---
whisper_model: WhisperModel | None = None
context_buffer: deque[tuple[float, str, str]] = deque()
context_lock = threading.Lock()

session_active: bool = False
live_insights_enabled: bool = False

# --- Audio buffer ---
audio_buffer: list[NDArray[np.float32]] = []
buffer_lock = threading.Lock()
MIN_BUFFER_LENGTH = 0.75
SAMPLE_RATE = 16000
VAD_FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)
FRAME_BYTES = FRAME_SAMPLES * 2
vad = webrtcvad.Vad(2)


# --- Audio Helpers ---
def is_speech(audio: NDArray[np.float32]) -> bool:
    """Check if audio contains speech using VAD."""
    try:
        if len(audio) == 0:
            return False
        int16_bytes = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        if len(int16_bytes) < FRAME_BYTES:
                   return False
        for offset in range(0, len(int16_bytes) - (len(int16_bytes) % FRAME_BYTES), FRAME_BYTES):
                frame = int16_bytes[offset:offset + FRAME_BYTES]
                if vad.is_speech(frame, SAMPLE_RATE):  # pyright: ignore[reportUnknownMemberType]
                    return True
        return False
    except Exception as e:
        logging.error(f"Error checking speech: {e}")
        return False

def flush_audio_buffer() -> NDArray[np.float32] | None:
    global audio_buffer
    with buffer_lock:
        if not audio_buffer:
            return None
        combined: NDArray[np.float32] = np.concatenate(audio_buffer)
        audio_buffer = []
        return combined


# --- Context ---
def add_to_context(text_type: str, text:str) -> None:
    with context_lock:
            ts = time.time()
            context_buffer.append((ts, text_type, text))
            current_length = sum(len(item[2]) for item in context_buffer)
            while current_length > CONTEXT_BUFFER_MAX_LENGTH and len(context_buffer) > 1:
                removed = context_buffer.popleft()
                current_length -= len(removed[2])
            while context_buffer and (time.time() - context_buffer[0][0] > CONTEXT_BUFFER_MAX_TIME_SECONDS):
                context_buffer.popleft()  # pyright: ignore[reportUnusedCallResult]

def get_current_context() -> str:
    with context_lock:
            return "\n".join(item[2] for item in context_buffer)

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
    base_instructions = """You are Cue, an AI assistant for neurodivergent developers.
    - Provide clear, structured answers
    - Use step-by-step explanations
    - Reduce anxiety with supportive tone
    - Offer focus & executive function help"""

    style_instructions = ""
    if style:
            style = style.upper()
            if style == "STAR":
                style_instructions = "\nUse STAR format: Situation, Task, Action, Result."
            elif style == "PAR":
                style_instructions = "\nUse PAR format: Problem, Action, Result."
            elif style == "SCQA":
                style_instructions = "\nUse SCQA: Situation, Complication, Question, Answer."

    return f"""{base_instructions}{style_instructions}

    Context type: {context_type}
    Context: {context}

    User query: "{query}"

    Respond concisely, supportively, and clearly."""


# --- Whisper ---
def load_whisper_model() -> None:
    global whisper_model
    try:
        whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        logging.info("Whisper model loaded.")
    except Exception as e:
        logging.error(f"Failed to load Whisper: {e}")
        whisper_model = None

# --- FastAPI Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):  # pyright: ignore[reportUnusedParameter]
    threading.Thread(target=load_whisper_model, daemon=True).start()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "ok", "whisper_loaded": whisper_model is not None}


# --- WebSocket ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global session_active
    await websocket.accept()
    logging.info("WebSocket connected.")
    try:
        while True:
            msg = json.loads(await websocket.receive_text())  # pyright: ignore[reportAny]
            msg_type = msg.get("type")  # pyright: ignore[reportAny]

            if msg_type == "audio_chunk":
                await handle_audio_chunk(websocket, msg)  # pyright: ignore [reportAny]
            elif msg_type == "image_data":
                await handle_image_data(websocket, msg)  # pyright: ignore[ reportAny]
            elif msg_type == "llm_query":
                await handle_llm_query(websocket, msg)  # pyright: ignore[reportAny]
            elif msg_type == "start_session":
                session_active = True
                await websocket.send_json({"type": "session_status", "status": "started"})
            elif msg_type == "stop_session":
                session_active = False
                await websocket.send_json({"type": "session_status", "status": "stopped"})
            else:
                await websocket.send_json({"status": "error", "message": f"Unknown type: {msg_type}"})
    except WebSocketDisconnect:
        logging.info("WebSocket disconnected.")


# --- Handlers ---
async def handle_audio_chunk(websocket: WebSocket, message: dict[str, Any]) -> None:  # pyright: ignore[reportExplicitAny]
    if not whisper_model:
        await websocket.send_json({"type": "transcript_error", "message": "Whisper not loaded."})
        return
    try:
        audio_bytes = base64.b64decode(message.get("data"), validate=True)  # pyright: ignore[reportArgumentType]
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            pcm16: NDArray[np.int16] = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            audio_np: NDArray[np.float32] = pcm16.astype(np.float32) / 32768.0
        if not is_speech(audio_np):
            return
        with buffer_lock:
            audio_buffer.append(audio_np)
            total_duration = sum(len(buf) for buf in audio_buffer) / SAMPLE_RATE
        if total_duration >= MIN_BUFFER_DURATION:  # pyright: ignore[reportUndefinedVariable]
            combined = flush_audio_buffer()
            if combined is None or len(combined) == 0:
                return
            segments, info = whisper_model.transcribe(combined, beam_size=5)  # pyright: ignore[reportUnusedVariable, reportUnknownMemberType]
            transcript = "".join(s.text for s in segments)
            if transcript.strip():
                add_to_context("audio", transcript)
                await websocket.send_json({"type": "transcript", "text": transcript})
                if session_active:
                    asyncio.create_task(auto_llm_suggestion(websocket, transcript))  # pyright: ignore[reportUnusedCallResult]
    except Exception as e:
        logging.error(f"Audio error: {e}", exc_info=True)
        await websocket.send_json({"type": "transcript_error", "message": str(e)})


async def handle_image_data(websocket: WebSocket, message: dict[str, Any]) -> None:  # pyright: ignore[reportExplicitAny]
    try:
        image = Image.open(io.BytesIO(base64.b64decode(message.get("data")))).convert("RGB")  # pyright: ignore[reportArgumentType]
        extracted = pytesseract.image_to_string(image)  # pyright: ignore[reportUnknownMemberType]
        if extracted.strip():  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            add_to_context("screen", extracted)  # pyright: ignore[reportArgumentType]
            await websocket.send_json({"type": "ocr_result", "text": extracted})
            if session_active:
                asyncio.create_task(auto_llm_suggestion(websocket, extracted))  # pyright: ignore[reportUnusedCallResult, reportArgumentType]
    except Exception as e:
        logging.error(f"OCR error: {e}", exc_info=True)
        await websocket.send_json({"type": "ocr_error", "message": str(e)})


async def handle_llm_query(websocket: WebSocket, message: dict[str, Any]) -> None:  # pyright: ignore[reportExplicitAny]
    query: str = message.get("query", "")  # pyright: ignore[reportAny]
    if not query.strip():
        await websocket.send_json({"type": "llm_response_error", "message": "Empty query."})
        return
    context = get_current_context()
    prompt = build_neurodivergent_prompt(context, query, detect_context_type(context), message.get("style"))
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            headers={"Content-Type": "application/json"},
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
            stream=True,
        )
        resp.raise_for_status()
        chunks: list[str] = []
        for chunk in resp.iter_lines():  # pyright: ignore[reportAny]
            if chunk:
                try:
                    j = json.loads(chunk.decode("utf-8"))  # pyright: ignore[reportAny]
                    token = j.get("response", "")  # pyright: ignore[reportAny]
                    if token:
                        chunks.append(token)  # pyright: ignore[reportAny]
                        await websocket.send_json({"type": "llm_response_chunk", "text_chunk": token})
                    if j.get("done"):  # pyright: ignore[reportAny]
                        break
                except json.JSONDecodeError:
                    continue
        full = "".join(chunks)
        if full.strip():
            add_to_context("llm_response", full)
            await websocket.send_json({"type": "llm_response_complete", "text": full})
    except Exception as e:
        logging.error(f"LLM error: {e}", exc_info=True)
        await websocket.send_json({"type": "llm_response_error", "message": str(e)})


async def auto_llm_suggestion(websocket: WebSocket, latest_text: str) -> None:
    context = get_current_context()
    prompt = build_neurodivergent_prompt(context, f"Give a short hint: {latest_text}", detect_context_type(context))
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            headers={"Content-Type": "application/json"},
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
            stream=True,
        )
        resp.raise_for_status()
        chunks: list[str] = []
        for chunk in resp.iter_lines():  # pyright: ignore[reportAny]
            if chunk:
                try:
                    j = json.loads(chunk.decode("utf-8"))  # pyright: ignore[reportAny]
                    token = j.get("response", "")  # pyright: ignore[reportAny]
                    if token:
                        chunks.append(token)  # pyright: ignore[reportAny]
                        await websocket.send_json({"type": "auto_hint_chunk", "text_chunk": token})
                    if j.get("done"):  # pyright: ignore[reportAny]
                        break
                except json.JSONDecodeError:
                    continue
        full = "".join(chunks)
        if full.strip():
            add_to_context("auto_hint", full)
            await websocket.send_json({"type": "auto_hint_complete", "text": full})
    except Exception as e:
        logging.error(f"Auto-hint error: {e}", exc_info=True)
        await websocket.send_json({"type": "auto_hint_error", "message": str(e)})


if __name__ == "__main__":
    import uvicorn
    logging.info(f"Starting backend on {OLLAMA_HOST} with {OLLAMA_MODEL}")
    uvicorn.run(app, host="127.0.0.1", port=8001, ws="websockets")
