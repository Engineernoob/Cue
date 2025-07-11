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
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import numpy as np
from pytesseract import image_to_string, pytesseract
import requests
import sounddevice as sd

# --- Configuration ---
# Configure logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to Tesseract executable (adjust if necessary for your OS)
# On Windows, it might be something like: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# On Linux/macOS, it's usually in PATH if installed, but you can specify:
# pytesseract.tesseract_cmd = '/usr/local/bin/tesseract' # Example for macOS Homebrew
# pytesseract.tesseract_cmd = '/usr/bin/tesseract' # Example for Linux
# Ensure Tesseract is installed and its path is correctly set if not in system PATH.
# Example: pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Ollama server configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest") # Or any other model you have pulled in Ollama

# Faster Whisper model configuration
# Options: tiny, base, small, medium, large, large-v2, large-v3
# For better performance on CPU, consider 'tiny' or 'base'.
# For GPU, 'small' or 'medium' might be good. 'large' models require significant VRAM.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu") # "cpu" or "cuda"
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8") # "int8" for CPU, "float16" for GPU

# Context buffer settings
CONTEXT_BUFFER_MAX_LENGTH = 2000 # Max characters for the LLM context
CONTEXT_BUFFER_MAX_TIME_SECONDS = 300 # Max time (5 minutes) for context

# --- Global Variables ---
app = FastAPI()
whisper_model: Optional[WhisperModel] = None
context_buffer = deque() # Stores (timestamp, text_type, text) tuples for context
context_lock = threading.Lock() # To safely access context_buffer from multiple threads

# --- Helper Functions ---
def add_to_context(text_type: str, text: str):
    """Adds text to the rolling context buffer."""
    with context_lock:
        timestamp = time.time()
        context_buffer.append((timestamp, text_type, text))

        # Trim by length
        current_length = sum(len(item[2]) for item in context_buffer)
        while current_length > CONTEXT_BUFFER_MAX_LENGTH and len(context_buffer) > 1:
            removed_item = context_buffer.popleft()
            current_length -= len(removed_item[2])

        # Trim by time
        while context_buffer and (time.time() - context_buffer[0][0] > CONTEXT_BUFFER_MAX_TIME_SECONDS):
            context_buffer.popleft()

def get_current_context() -> str:
    """Retrieves the current aggregated context as a single string."""
    with context_lock:
        context_texts = [item[2] for item in context_buffer]
        return "\n".join(context_texts)

def load_whisper_model():
    """Loads the Whisper model into memory."""
    global whisper_model
    try:
        logging.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE} on {WHISPER_DEVICE} with {WHISPER_COMPUTE_TYPE} compute type...")
        whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        logging.info("Whisper model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load Whisper model: {e}")
        whisper_model = None # Ensure it's None if loading fails

# --- FastAPI Endpoints ---
@app.on_event("startup")
async def startup_event():
    """Event handler for application startup."""
    # Load Whisper model in a separate thread to not block startup
    threading.Thread(target=load_whisper_model).start()

@app.get("/health")
async def health_check():
    """Endpoint to check if the backend is running."""
    return {"status": "ok", "whisper_loaded": whisper_model is not None}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication with Electron."""
    await websocket.accept()
    logging.info("WebSocket connection established.")

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
            else:
                logging.warning(f"Unknown message type: {msg_type}")
                await websocket.send_json({"status": "error", "message": f"Unknown type: {msg_type}"})

    except WebSocketDisconnect:
        logging.info("WebSocket connection disconnected.")
    except Exception as e:
        logging.error(f"WebSocket error: {e}", exc_info=True)
        # Attempt to send an error message before closing
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
        except RuntimeError:
            pass # WebSocket already closed

async def handle_audio_chunk(websocket: WebSocket, message: Dict[str, Any]):
    """Handles incoming audio chunks for real-time STT."""
    if not whisper_model:
        await websocket.send_json({"type": "transcript_error", "message": "Whisper model not loaded."})
        return

    try:
        audio_bytes_b64 = message.get("data")
        if not audio_bytes_b64:
            logging.warning("Received empty audio_chunk data.")
            return

        audio_bytes = base64.b64decode(audio_bytes_b64)
        # Convert bytes to a numpy array (assuming 16kHz, float32, mono)
        # Adjust dtype and sample rate based on your Electron capture settings
        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)

        # Transcribe the audio chunk
        # Note: For continuous real-time, you might need a more sophisticated
        # buffering/segmentation strategy than transcribing each small chunk directly.
        # faster-whisper's transcribe method expects a path or a numpy array.
        # For true streaming, you'd feed chunks into a buffer and process when
        # enough audio is accumulated or silence is detected.
        # For simplicity, this example transcribes each chunk.
        segments, info = whisper_model.transcribe(audio_np, beam_size=5)

        full_transcript = ""
        for segment in segments:
            full_transcript += segment.text

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
    """Handles incoming image data for OCR."""
    try:
        image_bytes_b64 = message.get("data")
        if not image_bytes_b64:
            logging.warning("Received empty image_data.")
            return

        image_bytes = base64.b64decode(image_bytes_b64)
        image = Image.open(io.BytesIO(image_bytes))

        # Perform OCR
        extracted_text = pytesseract.image_to_string(image)

        if extracted_text.strip():
            add_to_context("screen", extracted_text)
            await websocket.send_json({
                "type": "ocr_result",
                "text": extracted_text,
                "timestamp": time.time()
            })
            logging.info(f"OCR Result: {extracted_text[:100]}...") # Log first 100 chars
    except Exception as e:
        logging.error(f"Error processing image data for OCR: {e}", exc_info=True)
        await websocket.send_json({"type": "ocr_error", "message": str(e)})

async def handle_llm_query(websocket: WebSocket, message: Dict[str, Any]):
    """Handles LLM queries and streams responses from Ollama."""
    user_query = message.get("query", "")
    trigger_type = message.get("trigger", "manual") # e.g., "manual", "hotkey", "contextual"

    if not user_query.strip():
        await websocket.send_json({"type": "llm_response_error", "message": "Empty query received."})
        return

    current_context = get_current_context()

    # Construct the prompt for Ollama
    # This prompt can be refined based on desired assistant behavior
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
        # Stream response from Ollama
        ollama_api_url = f"{OLLAMA_HOST}/api/generate"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True, # Request streaming from Ollama
            "options": {
                "temperature": 0.7,
                "num_predict": 200 # Max tokens to generate
            }
        }

        response_chunks = []
        with requests.post(ollama_api_url, headers=headers, json=payload, stream=True) as response:
            response.raise_for_status() # Raise an exception for HTTP errors
            for chunk in response.iter_lines():
                if chunk:
                    try:
                        json_chunk = json.loads(chunk.decode('utf-8'))
                        token = json_chunk.get("response", "")
                        done = json_chunk.get("done", False)

                        if token:
                            response_chunks.append(token)
                            # Send partial response to Electron for real-time display
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
            add_to_context("llm_response", full_llm_response) # Add LLM's response to context
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

# --- Main execution for Uvicorn ---
if __name__ == "__main__":
    import uvicorn
    # To run: uvicorn main:app --host 127.0.0.1 --port 8000 --ws websockets
    # Ensure Tesseract is in your system's PATH or set pytesseract.tesseract_cmd above.
    logging.info(f"Starting FastAPI backend. Ollama Host: {OLLAMA_HOST}, Model: {OLLAMA_MODEL}")
    logging.info(f"Whisper Model: {WHISPER_MODEL_SIZE}, Device: {WHISPER_DEVICE}, Compute Type: {WHISPER_COMPUTE_TYPE}")
    uvicorn.run(app, host="127.0.0.1", port=8000, ws="websockets")
