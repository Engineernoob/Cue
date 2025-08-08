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
from typing import Any, Dict, Optional

from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import numpy as np
from pytesseract import image_to_string, pytesseract
import requests

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

CONTEXT_BUFFER_MAX_LENGTH = 2000
CONTEXT_BUFFER_MAX_TIME_SECONDS = 300

app = FastAPI()
whisper_model: Optional[WhisperModel] = None
context_buffer = deque()
context_lock = threading.Lock()

# --- Session and Live Insights State ---
session_active = False
live_insights_enabled = False

def add_to_context(text_type: str, text: str):
    with context_lock:
        timestamp = time.time()
        context_buffer.append((timestamp, text_type, text))

        current_length = sum(len(item[2]) for item in context_buffer)
        while current_length > CONTEXT_BUFFER_MAX_LENGTH and len(context_buffer) > 1:
            removed_item = context_buffer.popleft()
            current_length -= len(removed_item[2])

        while context_buffer and (time.time() - context_buffer[0][0] > CONTEXT_BUFFER_MAX_TIME_SECONDS):
            context_buffer.popleft()

def get_current_context() -> str:
    with context_lock:
        context_texts = [item[2] for item in context_buffer]
        return "\n".join(context_texts)

def load_whisper_model():
    global whisper_model
    try:
        logging.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE} on {WHISPER_DEVICE} with {WHISPER_COMPUTE_TYPE} compute type...")
        whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        logging.info("Whisper model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load Whisper model: {e}")
        whisper_model = None

async def send_live_insights_periodically(websocket: WebSocket):
    while live_insights_enabled:
        context = get_current_context()
        summary = context[-500:]  # last 500 chars as a simple summary snippet
        try:
            await websocket.send_json({
                "type": "live_summary",
                "text": summary,
                "timestamp": time.time()
            })
        except Exception:
            break
        await asyncio.sleep(5)

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=load_whisper_model).start()

@app.get("/health")
async def health_check():
    return {"status": "ok", "whisper_loaded": whisper_model is not None}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global session_active, live_insights_enabled

    await websocket.accept()
    logging.info("WebSocket connection established.")

    live_insights_task = None

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
                session_active = True
                logging.info("Session started.")
                await websocket.send_json({"type": "session_status", "status": "started"})

            elif msg_type == "stop_session":
                session_active = False
                logging.info("Session stopped.")
                await websocket.send_json({"type": "session_status", "status": "stopped"})

            elif msg_type == "toggle_live_insights":
                live_insights_enabled = not live_insights_enabled
                logging.info(f"Live insights toggled: {live_insights_enabled}")

                if live_insights_enabled:
                    live_insights_task = asyncio.create_task(send_live_insights_periodically(websocket))
                else:
                    if live_insights_task:
                        live_insights_task.cancel()
                        live_insights_task = None

                await websocket.send_json({"type": "live_insights_status", "enabled": live_insights_enabled})

            else:
                logging.warning(f"Unknown message type: {msg_type}")
                await websocket.send_json({"status": "error", "message": f"Unknown type: {msg_type}"})

    except WebSocketDisconnect:
        logging.info("WebSocket disconnected.")
        if live_insights_task:
            live_insights_task.cancel()
    except Exception as e:
        logging.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
        except RuntimeError:
            pass

async def handle_audio_chunk(websocket: WebSocket, message: Dict[str, Any]):
    if not whisper_model:
        await websocket.send_json({"type": "transcript_error", "message": "Whisper model not loaded."})
        return

    try:
        audio_bytes_b64 = message.get("data")
        if not audio_bytes_b64:
            logging.warning("Received empty audio_chunk data.")
            return

        audio_bytes = base64.b64decode(audio_bytes_b64)

        # 🚨 Buffer safety check before converting to float32
        if len(audio_bytes) % 4 != 0:
            logging.warning(f"Skipping malformed audio chunk (size={len(audio_bytes)} not divisible by 4)")
            return

        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)

        segments, info = whisper_model.transcribe(audio_np, beam_size=5)

        full_transcript = "".join(segment.text for segment in segments)

        if full_transcript.strip():
            add_to_context("audio", full_transcript)
            await websocket.send_json({
                "type": "transcript",
                "text": full_transcript,
                "language": info.language,
                "probability": info.language_probability,
                "timestamp": time.time()
            })
            logging.info(f"Transcribed: {full_transcript}")

    except Exception as e:
        logging.error(f"Error processing audio chunk: {e}", exc_info=True)
        await websocket.send_json({"type": "transcript_error", "message": str(e)})


async def handle_image_data(websocket: WebSocket, message: Dict[str, Any]):
    try:
        image_bytes_b64 = message.get("data")
        if not image_bytes_b64:
            logging.warning("Received empty image_data.")
            return

        image_bytes = base64.b64decode(image_bytes_b64)

        # 🛡️ Safe load + format conversion to prevent format-related issues
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as img_error:
            logging.warning(f"Invalid image format or unreadable image: {img_error}")
            await websocket.send_json({
                "type": "ocr_error",
                "message": "Unreadable or unsupported image format."
            })
            return

        extracted_text = pytesseract.image_to_string(image)

        if extracted_text.strip():
            add_to_context("screen", extracted_text)
            await websocket.send_json({
                "type": "ocr_result",
                "text": extracted_text,
                "timestamp": time.time()
            })
            logging.info(f"OCR Result: {extracted_text[:100]}...")

    except Exception as e:
        logging.error(f"Error processing image data for OCR: {e}", exc_info=True)
        await websocket.send_json({"type": "ocr_error", "message": str(e)})


async def handle_llm_query(websocket: WebSocket, message: Dict[str, Any]):
    user_query = message.get("query", "")
    trigger_type = message.get("trigger", "manual")

    if not user_query.strip():
        await websocket.send_json({"type": "llm_response_error", "message": "Empty query received."})
        return

    current_context = get_current_context()

    prompt = f"""You are a helpful AI desktop assistant.
You are currently observing the user's screen and listening to their audio.
Here is the recent context from the user's environment:
---
{current_context}
---

The user has triggered you with the following query: "{user_query}"

Provide a concise, context-aware response or coaching. If the context is not directly relevant to the query, state that you don't have enough context but still try to provide a general helpful response.
"""
    logging.info(f"Sending LLM query to Ollama. Context length: {len(current_context)}. Query: {user_query}")

    try:
        ollama_api_url = f"{OLLAMA_HOST}/api/generate"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "num_predict": 200
            }
        }

        response_chunks = []
        with requests.post(ollama_api_url, headers=headers, json=payload, stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_lines():
                if chunk:
                    try:
                        json_chunk = json.loads(chunk.decode('utf-8'))
                        token = json_chunk.get("response", "")
                        done = json_chunk.get("done", False)

                        if token:
                            response_chunks.append(token)
                            await websocket.send_json({
                                "type": "llm_response_chunk",
                                "text_chunk": token
                            })
                        if done:
                            break
                    except json.JSONDecodeError:
                        logging.warning(f"Could not decode JSON chunk from Ollama: {chunk}")
                        continue

        full_llm_response = "".join(response_chunks)
        if full_llm_response.strip():
            add_to_context("llm_response", full_llm_response)
            await websocket.send_json({
                "type": "llm_response_complete",
                "text": full_llm_response,
                "timestamp": time.time()
            })
            logging.info(f"LLM Response Complete: {full_llm_response}")

    except requests.exceptions.ConnectionError as e:
        logging.error(f"Could not connect to Ollama server at {OLLAMA_HOST}: {e}")
        await websocket.send_json({"type": "llm_response_error", "message": "Could not connect to Ollama server. Please ensure Ollama is running."})
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Ollama API: {e}", exc_info=True)
        await websocket.send_json({"type": "llm_response_error", "message": f"Error from Ollama: {e}"})
    except Exception as e:
        logging.error(f"Unexpected error during LLM query: {e}", exc_info=True)
        await websocket.send_json({"type": "llm_response_error", "message": f"An unexpected error occurred: {e}"})

if __name__ == "__main__":
    import uvicorn
    logging.info(f"Starting FastAPI backend. Ollama Host: {OLLAMA_HOST}, Model: {OLLAMA_MODEL}")
    logging.info(f"Whisper Model: {WHISPER_MODEL_SIZE}, Device: {WHISPER_DEVICE}, Compute Type: {WHISPER_COMPUTE_TYPE}")
    uvicorn.run(app, host="127.0.0.1", port=8000, ws="websockets")
