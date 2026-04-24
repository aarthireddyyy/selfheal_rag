# Self-Healing RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) system that automatically grades its own answers and retries with rephrased queries when grading fails — preventing hallucinations by returning an honest "I don't know" instead of making things up.

## What Problem Does This Solve?

Basic RAG systems retrieve documents and generate answers but have no quality control. They hallucinate confidently, giving wrong answers with no fallback.

This system is smarter. After generating an answer, it:
1. Grades whether the answer is grounded in the retrieved documents
2. If grading fails → rewrites the question and retries (max 2 retries)
3. If all retries fail → returns "I don't know" instead of hallucinating

Real-world use case: companies building knowledge base chatbots over internal documents (HR policies, product manuals, legal contracts) where hallucination destroys user trust.

---

## Architecture

```
User Question
     ↓
[Retrieve] → top-4 chunks from ChromaDB
     ↓
[Generate] → Groq Llama 3.3 answers using only those chunks
     ↓
[Grade] → Groq judges: is this answer supported by the chunks?
     ↓
  PASS → return answer
  FAIL → [Rewrite] → new query → back to Retrieve (max 2x)
  STILL FAILING → "I don't know"
```

The retry loop is orchestrated by LangGraph as an explicit state machine.

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Language | Python 3.12+ | Modern async support, type hints |
| Package Manager | uv | 10-100x faster than pip |
| Vector Database | ChromaDB | Local, no server needed, persists to disk |
| Embeddings | sentence-transformers | Free, runs locally, no API calls |
| LLM | Groq Llama 3.3 70B | Free tier, very fast inference |
| RAG Framework | LangChain + LangGraph | State machine for self-healing loop |
| API Framework | FastAPI | Auto-validation, auto-docs, async-ready |
| Chunking | SemanticChunker | Splits by meaning, not fixed size |

---

## Project Structure

```
self-healing-rag/
├── src/
│   ├── ingest.py          # Document ingestion pipeline
│   ├── rag_agent.py       # LangGraph self-healing agent
│   └── api.py             # FastAPI REST API layer
├── docs/                  # Your knowledge base (.txt, .pdf files)
├── chroma_db/             # Vector store (auto-created)
├── postman/               # Postman collection for API testing
├── .env                   # Environment variables (GROQ_API_KEY)
├── pyproject.toml         # Dependencies
├── Dockerfile             # Docker image definition
└── docker-compose.yml     # Docker orchestration
```

---

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- A free [Groq API key](https://console.groq.com)

### Installation

1. Clone the repo and navigate to the project directory

2. Install dependencies:
```bash
uv sync
```

3. Create a `.env` file:
```bash
GROQ_API_KEY=your_groq_api_key_here
CHROMA_PERSIST_DIR=./chroma_db
```

4. Add your documents to the `docs/` folder (supports `.txt` and `.pdf`)

5. Run ingestion to load documents into ChromaDB:
```bash
uv run python src/ingest.py
```

---

## Usage

### Option 1: Interactive CLI

```bash
uv run python src/rag_agent.py
```

Ask questions interactively. Type `quit` to exit.

### Option 2: REST API

Start the server:
```bash
uv run uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Then:
- Open `http://localhost:8000/docs` for interactive Swagger UI
- Or use curl:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main application domains of Generative AI?"}'
```

## API Endpoints

### `POST /query`

Ask a question against the knowledge base.

**Request:**
```json
{
  "question": "What is Generative AI?"
}
```

**Response:**
```json
{
  "answer": "Generative AI refers to artificial intelligence that can generate novel content...",
  "sources": ["docs/survey.pdf", "docs/survey.pdf"],
  "attempts": 1
}
```

- `attempts=1` → answer passed grading on first try
- `attempts=3` → exhausted all retries, returned "I don't know"

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

---

## License

MIT

---

## Acknowledgments

Built with:
- [LangChain](https://github.com/langchain-ai/langchain) for RAG primitives
- [LangGraph](https://github.com/langchain-ai/langgraph) for state machine orchestration
- [ChromaDB](https://github.com/chroma-core/chroma) for vector storage
- [FastAPI](https://github.com/tiangolo/fastapi) for the API layer
- [Groq](https://groq.com) for fast LLM inference
