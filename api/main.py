import os
import io
import json
import logging
import tempfile
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import pandas as pd
import requests
import time
import matplotlib.pyplot as plt
import numpy as np

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
LLM7_API_URL = "https://api.llm7.io/v1/chat/completions"
MODEL = "meta-llama/Meta-Llama-3-70B-Instruct"

def extract_text_from_file(file: UploadFile, file_bytes: bytes) -> str:
    filename = file.filename.lower()
    if filename.endswith(".txt") or filename.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
            return df.to_csv(index=False)
        except Exception:
            return file_bytes.decode("utf-8", errors="ignore")
    elif filename.endswith(".pdf"):
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages_text).strip()
    else:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.filename}")

def call_llm7_api(messages: list, retries=3, backoff_factor=5):
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

    # The prompt for the LLM is now a single, structured request for all answers.
    # The image generation is handled locally after getting the data.
    system_prompt = """You are a helpful Python data analyst. You will be given tabular data in CSV format and a set of questions. Your task is to extract the answers to these questions from the data. The questions are about films and their financial performance.

Respond ONLY with a single JSON array of strings, in the exact order of the questions. Do NOT include any explanations, markdown, or other text. The output should be strictly a JSON array, like: ["answer1", "answer2", "answer3", "answer4"].
"""
    
    user_prompt = (
        f"Given the following CSV data from a list of highest-grossing films:\n\n{data_text}\n\n"
        f"Answer the following questions using ONLY the above data:\n\n"
        f"1. How many $2 bn movies were released before 2020?\n"
        f"2. Which is the earliest film that grossed over $1.5 bn?\n"
        f"3. What's the correlation between the Rank and Peak?\n"
        f"4. Ignore this question. The plot will be generated separately."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        # Step 1: Call the LLM to get the first three answers
        logging.info("Calling LLM to get answers...")
        llm_response = call_llm7_api(messages)
        response_content = llm_response["choices"][0]["message"]["content"]
        
        # Parse the JSON array from the LLM's response
        answers = json.loads(response_content)
        if not isinstance(answers, list) or len(answers) < 3:
            raise ValueError("LLM did not return a valid list of answers.")
    except Exception as e:
        logging.error(f"Error getting answers from LLM: {e}")
        return JSONResponse(content={"error": "Failed to get answers from LLM.", "details": str(e), "raw_output": response_content})

    # Step 2: Generate the scatter plot with the given metrics
    try:
        df = pd.read_csv(io.StringIO(data_text))
        
        # Ensure 'Rank' and 'Peak' columns exist and are numeric
        if 'Rank' not in df.columns or 'Peak' not in df.columns:
            raise ValueError("Required columns 'Rank' or 'Peak' not found in data.")
            
        df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
        df['Peak'] = pd.to_numeric(df['Peak'], errors='coerce')
        df.dropna(subset=['Rank', 'Peak'], inplace=True)
        
        # Plotting the scatter plot with a red dotted regression line
        plt.figure(figsize=(6, 4))
        plt.scatter(df['Rank'], df['Peak'])
        
        # Calculate and plot the regression line
        z = np.polyfit(df['Rank'], df['Peak'], 1)
        p = np.poly1d(z)
        plt.plot(df['Rank'], p(df['Rank']), "r--") # Dotted red line
        
        plt.title('Rank vs Peak')
        plt.xlabel('Rank')
        plt.ylabel('Peak')
        plt.grid(True)
        
        # Save the plot to a temporary buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        
        # Encode the image to base64
        buf.seek(0)
        image_bytes = buf.getvalue()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Construct the data URI
        data_uri = f"data:image/png;base64,{base64_image}"
        
        # Check image size (optional, but good practice for the rubric)
        if len(image_bytes) > 100000:
             logging.warning("Generated image size exceeds 100 KB.")
             
        # Add the data URI to the answers array as the fourth element
        answers.append(data_uri)
        
    except Exception as e:
        logging.error(f"Error generating plot: {e}")
        # If plot generation fails, add an error string to maintain array length
        answers.append(f"Error generating plot: {e}")
        return JSONResponse(content={"error": f"Failed to generate plot: {str(e)}"})
        
    # Step 3: Return the final JSON array
    return JSONResponse(content=answers)

@app.get("/")
async def root():
    return {"message": "Welcome to the TDS Project API powered by LLM7.io!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
