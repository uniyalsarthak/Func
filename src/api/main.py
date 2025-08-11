import sys
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rag.rag_pipeline import retrieve, generate
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# --- CORS Middleware ---
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    answer: str

# --- In-Memory Cache ---
cache = {}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Receives a chat query and returns the chatbot's answer.
    """
    try:
        # Check cache first
        if request.query in cache:
            print("   - Returning cached response.")
            return ChatResponse(answer=cache[request.query])

        context = await retrieve(request.query)
        if context:
            answer = await generate(request.query, context)
            cache[request.query] = answer  # Store in cache
            return ChatResponse(answer=answer)
        else:
            return ChatResponse(answer="Sorry, I couldn't find an answer.")
    except Exception as e:
        return ChatResponse(answer=f"An error occurred: {str(e)}")

class LogRequest(BaseModel):
    query: str
    answer: str
    feedback: str  # "up" or "down"

@app.post("/api/log_chat")
async def log_chat(request: LogRequest):
    """
    Logs a chat interaction to a file for fine-tuning.
    """
    log_file_path = os.path.join(os.path.dirname(__file__), '../data/chat_logs.jsonl')
    with open(log_file_path, "a") as f:
        f.write(f"{request.model_dump_json()}\n")
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Admissions Chatbot API"}

# To run this API, use the command:
# uvicorn src.api.main:app --reload
