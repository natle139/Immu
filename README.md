---
title: Immu
emoji: 🛂
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
short_description: Australian bridging visa Q&A from official documents
---

# Immu

Bridging visa & visa pathway navigator | RAG system

Ask plain-English questions about Australian bridging visas — work rights, travel, review options — answered from official Department of Home Affairs documents.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your Groq API key:

```
GROQ_API_KEY=your_key_here
```

## Ingest documents

Put source PDFs in `data/raw/`, then build the vector store (one-time, re-run when documents change):

```bash
python -m src.ingest
```

## Run

```bash
python app.py
```

Open http://localhost:8000 to ask questions, or POST to the API directly:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What can I work on with a Bridging Visa A?"}'
```

## Models

- **Generation:** `openai/gpt-oss-20b` via Groq
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local)

## Eval

`eval/golden.json` has labeled questions with expected source docs and answer keywords.
Run against the live retriever + Groq:

```bash
python -m eval.run_eval
```

Prints a pass/fail per question and a retrieval/answer score; exits non-zero below 70%.
