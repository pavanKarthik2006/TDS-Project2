from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
import io
import pandas as pd
from PyPDF2 import PdfReader
import requests
import logging
import time
import json
from typing import Any

load_dotenv()

app = FastAPI()

# CORS for frontend/tester access (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

PROMPT_PATH = os.path.join("prompts", "abdul_task_breakdown.txt")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB max per file
LLM7_API_URL = "https://api.llm7.io/v1/chat/completions"
MODEL = "meta-llama/Meta-Llama-3-70B-Instruct"

def extract_text_from_file(file: UploadFile, file_bytes: bytes) -> str:
    filename = file.filename.lower()
    if filename.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")
    elif filename.endswith(".csv"):
        return file_bytes.decode("utf-8", errors="ignore")
    elif filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages_text).strip()
    else:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.filename}")

def call_llm7_api(messages, retries=3, backoff_factor=5) -> Any:
    headers = {"Content-Type": "application/json"}
    data = {"model": MODEL, "messages": messages}
    for attempt in range(retries):
        try:
            response = requests.post(
                LLM7_API_URL,
                headers=headers,
                json=data,
                timeout=120,
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                wait_time = backoff_factor * (attempt + 1)
                logging.warning(f"Rate limited by LLM7.io. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                response.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Request error: {e}")
            if attempt == retries - 1:
                raise HTTPException(status_code=503, detail=f"LLM7.io API unavailable: {e}")
            time.sleep(backoff_factor * (attempt + 1))
    raise HTTPException(status_code=503, detail="Failed to get response from LLM7.io after retries")

def load_system_prompt() -> str:
    try:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.error(f"Failed to read system prompt file: {e}")
        return ""

@app.post("/api/")
async def analyze_files(data_file: UploadFile = File(...), question_file: UploadFile = File(...)):
    allowed_types = {"text/plain", "text/csv", "application/pdf"}
    if data_file.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail=f"Unsupported data file type: {data_file.content_type}")
    if question_file.content_type != "text/plain":
        raise HTTPException(status_code=415, detail=f"Unsupported question file type: {question_file.content_type}")

    data_bytes = await data_file.read()
    question_bytes = await question_file.read()
    if len(data_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"Data file too large (max {MAX_FILE_SIZE} bytes).")
    if len(question_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"Question file too large (max {MAX_FILE_SIZE} bytes).")

    logging.info(f"Received data file: {data_file.filename}, size: {len(data_bytes)} bytes")
    logging.info(f"Received question file: {question_file.filename}, size: {len(question_bytes)} bytes")

    data_text = extract_text_from_file(data_file, data_bytes)
    question_text = question_bytes.decode("utf-8", errors="ignore")
    system_prompt = load_system_prompt()

    combined_prompt = (
        f"Given the following CSV data:\n{data_text}\n\n"
        f"Answer the following questions **using ONLY the above data**:\n{question_text}\n\n"
        "Respond ONLY with a JSON array in this format:\n"
        "[<count_of_2bn_movies>, <earliest_film_title>, <correlation_float>, <optional_base64_image_or_null>]\n"
        "Do NOT include explanations or other text."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": combined_prompt},
    ]

    result_json = call_llm7_api(messages)
    stepwise_plan = result_json["choices"][0]["message"]["content"]

    try:
        response_array = json.loads(stepwise_plan)
        if not isinstance(response_array, list):
            raise ValueError("Response is not a JSON array.")
    except Exception as e:
        logging.warning(f"Failed to parse LLM output as JSON array: {e}")
        response_array = [stepwise_plan]

    return JSONResponse(content=response_array)

@app.get("/")
async def root():
    return {"message": "Welcome to the TDS Project API powered by LLM7.io!"}

# === Vercel integration for serverless deployment ===

# === Local development (can be omitted for Vercel deploy) ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
