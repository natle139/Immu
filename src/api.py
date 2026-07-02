from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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
    answer = generate_answer(query.query, documents, sources)
    return Answer(answer=answer, sources=sources)

@app.get("/health")
def health():
    return {"status": "ok"}

# ponytail: plain HTML+JS instead of a frontend framework, upgrade if the UI grows beyond one page
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Immu — Bridging Visa Navigator</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; }
  textarea { width: 100%; height: 80px; font-size: 16px; }
  button { margin-top: 8px; padding: 8px 16px; font-size: 16px; }
  #answer { white-space: pre-wrap; margin-top: 24px; line-height: 1.5; }
</style>
</head>
<body>
  <h1>Immu</h1>
  <p>Ask a question about your bridging visa.</p>
  <textarea id="q" placeholder="e.g. What can I work on with a Bridging Visa A?"></textarea><br>
  <button onclick="ask()">Ask</button>
  <div id="answer"></div>
<script>
async function ask() {
  const q = document.getElementById("q").value;
  const answerEl = document.getElementById("answer");
  if (!q.trim()) return;
  answerEl.textContent = "Thinking...";
  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q })
    });
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();
    answerEl.textContent = data.answer;
  } catch (e) {
    answerEl.textContent = "Something went wrong. Please try again.";
  }
}
</script>
</body>
</html>
"""