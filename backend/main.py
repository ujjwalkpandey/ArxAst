from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from engine import ArxivEngine

app = FastAPI()
engine = ArxivEngine()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    paper_id: str
    question: str

@app.get("/search")
async def search(q: str):
    return engine.search_arxiv(q)

@app.post("/index/{paper_id}")
async def index(paper_id: str):
    title = engine.fetch_and_index(paper_id)
    return {"status": "success", "title": title}

@app.post("/chat")
async def chat(request: ChatRequest):
    answer = engine.ask_question(request.paper_id, request.question)
    return {"answer": answer}