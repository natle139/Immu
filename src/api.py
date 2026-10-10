from fastapi import FastAPI, HTTPException
from groq import APIError
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_core.documents import Document
from src.retriever import build_retriever, retrieve_documents
from src.chain import generate_answer

app = FastAPI(title="Immu API", version="1.0")
vectorstore = build_retriever()

class Query(BaseModel):
    query: str

class Answer(BaseModel):
    answer: str
    sources: list[str]

@app.post("/ask", response_model=Answer)
def ask(query: Query):
    documents, sources = retrieve_documents(query.query, vectorstore)
    try:
        answer = generate_answer(query.query, documents, sources)
    except APIError:  # rate limit / quota / outage: the public demo shares one free-tier quota
        raise HTTPException(503, "Answer service unavailable")
    return Answer(answer=answer, sources=sources)

@app.get("/health")
def health():
    return {"status": "ok"}

# ponytail: plain HTML+JS instead of a frontend framework, upgrade if the UI grows beyond one page
@app.get("/")
def home():
    return FileResponse("frontend/index.html")
