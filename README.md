# Self-Healing RAG Pipeline

A production-grade Retrieval-Augmented Generation system that automatically detects and recovers from hallucinations before returning answers to users.

## The Problem

Standard RAG systems retrieve documents, generate an answer, and returgn it — with no quality control. They hallucinate confidently. This system grades every answer for groundedness and retries with a rephrased query when grading fails, returning an honest "I don't know" instead of fabricated information.

## Architecture

```
User Question
     ↓
[Guardrails] — blocks prompt injection, sanitizes input
     ↓
[Hybrid Retrieve] — BM25 + Vector Search → Reranker → top-4 chunks
     ↓
[Generate] — Groq Llama 3.3, answers using ONLY retrieved chunks
     ↓
[Grade] — LLM-as-judge: pass / partial / fail
     ↓
  pass    → return answer
  partial → return answer + transparency caveat
  fail    → rewrite query → retry (max 2x)
  fail×3  → "I don't know"
```

## What Makes This Different from Basic RAG

| Feature | Basic RAG | This System |
|---------|-----------|-------------|
| Retrieval | Vector search only | BM25 + Vector + Reranker |
| Quality control | None | LLM-as-judge grading |
| Hallucination handling | Returns bad answer | Retries with rephrased query |
| Partial answers | Silent | Flagged with caveat |
| Prompt injection | Vulnerable | Pattern matching + hardened prompts |
| Off-topic queries | Hallucinated answer | Reranker score gate → fast rejection |

## Tech Stack

- **LangGraph** — state machine orchestrating the self-healing retry loop
- **ChromaDB** — local vector database, persists to disk
- **sentence-transformers** — local embeddings, zero API cost
- **SemanticChunker** — splits documents by meaning boundaries, not fixed character counts
- **BM25 (rank-bm25)** — keyword search for hybrid retrieval
- **cross-encoder/ms-marco-MiniLM-L-6-v2** — reranker for chunk quality scoring
- **Groq Llama 3.3 70B** — generation + grading (free tier)
- **FastAPI** — async REST API with Pydantic validation
- **uv** — dependency management

## Project Structure

```
├── src/
│   ├── ingest.py       # Document ingestion: load → semantic chunk → embed → store
│   ├── retriever.py    # Hybrid retrieval: BM25 + vector + reranker
│   ├── rag_agent.py    # LangGraph state machine: retrieve → generate → grade → retry
│   ├── guardrails.py   # Input sanitization + prompt injection defense
│   └── api.py          # FastAPI: POST /query, GET /health
├── docs/               # Knowledge base (.txt, .pdf)
├── chroma_db/          # Vector store (auto-created)
├── postman/            # API test collection
├── Dockerfile
└── docker-compose.yml
```

## Setup

```bash
# Install dependencies
uv sync

# Configure environment
echo "GROQ_API_KEY=your_key_here" > .env
echo "CHROMA_PERSIST_DIR=./chroma_db" >> .env

# Add documents to docs/ (.txt or .pdf), then ingest
uv run python src/ingest.py

# Start API
uv run uvicorn src.api:app --reload --port 8000
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

## API

### `POST /query`
```json
// Request
{ "question": "What are the main application domains of Generative AI?" }

// Response
{
  "answer": "The main application domains include text, images, video...",
  "sources": ["docs/survey.pdf", "docs/survey.pdf"],
  "attempts": 1
}
```
`attempts=1` → passed grading first try. `attempts=3` → all retries exhausted, returned "I don't know".

### `GET /health`
```json
{ "status": "ok" }
```

## How Hybrid Retrieval Works

Pure vector search misses exact keyword matches (model names, technical terms). BM25 misses semantic matches. Combining both then reranking with a cross-encoder gives significantly better chunk quality before the LLM ever sees the context.

```
Query → Vector Search (top-10) + BM25 (top-10)
      → Merge + deduplicate (~20 candidates)
      → Cross-encoder reranker scores all candidates
      → Return top-4 by reranker score
      → Relevance gate: if top score < -5.0, return empty (fast rejection)
```

## How Semantic Chunking Works

Documents are split by meaning, not by fixed character counts. The `SemanticChunker` embeds each sentence, measures similarity between adjacent sentences, and splits where similarity drops — keeping related ideas together in the same chunk.

Fixed chunking splits mid-thought. Semantic chunking preserves context. Better chunks → better retrieval → fewer retries.

## Security

Three-layer prompt injection defense:
1. **Pattern matching** — 20+ regex patterns block known injection attempts before the LLM runs
2. **Hardened system prompt** — explicit instructions to ignore embedded instructions
3. **Reranker score gate** — off-topic/adversarial queries score below threshold and are rejected without LLM calls


## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | required | Groq API key |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector store path |
| `TRANSFORMERS_OFFLINE` | `1` | Skip HuggingFace Hub checks |
| `HF_HUB_OFFLINE` | `1` | Run embeddings fully offline |

## Known Limitations

- Single worker processes one request at a time (~500MB RAM per additional worker)
- Groq free tier: 30 requests/min (3 retries × 10 users = rate limit risk)
- Pattern-based injection detection can be bypassed by novel phrasing — a dedicated classifier would be more robust
- No conversation history — each query is stateless

## License

MIT
