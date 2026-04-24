"""
api.py — FastAPI REST API layer

This file wraps the RAG agent as an HTTP service.
Any client (browser, mobile app, another service) can now query
your knowledge base by sending a POST request to /query.

WHY FastAPI over Flask or Django?
- Automatic request/response validation via Pydantic models
- Auto-generated interactive docs at /docs (Swagger UI) — free
- Async-ready, fast, modern Python
- Type hints = IDE autocomplete everywhere

KEY CONCEPT — lifespan:
The agent takes ~10 seconds to initialize (embedding model + ChromaDB).
We do this ONCE at server startup using a lifespan context manager,
store it in app.state.agent, and reuse it for every request.
Without this, every request would wait 10 seconds. With it: milliseconds.
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.rag_agent import create_rag_agent, query_rag

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PYDANTIC MODELS — request/response contracts
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    """
    Defines what the client must send in the POST /query body.

    WHY Pydantic?
    FastAPI uses Pydantic to validate incoming JSON automatically.
    If 'question' is missing or empty, FastAPI returns HTTP 422
    with a clear error message — before your code even runs.
    You never write manual validation code.

    min_length=1 means empty strings are rejected. (Req 8.3)
    """
    question: str = Field(..., min_length=1, description="The question to ask the knowledge base")


class QueryResponse(BaseModel):
    """
    Defines exactly what the API returns.
    Pydantic ensures the response always matches this shape. (Req 8.2)
    """
    answer: str
    sources: list[str]
    attempts: int


# ─────────────────────────────────────────────
# LIFESPAN — startup and shutdown logic
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at server startup (before any requests) and once at shutdown.

    WHY asynccontextmanager?
    FastAPI's lifespan expects an async context manager.
    Everything before 'yield' runs at startup.
    Everything after 'yield' runs at shutdown.

    At startup we:
    1. Check GROQ_API_KEY exists — fail fast if missing (Req 9.2)
    2. Initialize the RAG agent (loads embedding model + connects to ChromaDB)
    3. Store it in app.state so every endpoint can access it

    This means the heavy initialization happens ONCE, not per-request.
    """
    # Req 9.2: fail fast if API key is missing
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file and restart."
        )

    logger.info("Starting up — initializing RAG agent...")
    app.state.agent = create_rag_agent()
    logger.info("RAG agent ready. Server accepting requests.")

    yield  # server is running — handle requests

    # Shutdown cleanup (nothing needed here, but the pattern requires it)
    logger.info("Shutting down.")


# ─────────────────────────────────────────────
# APP INSTANCE
# ─────────────────────────────────────────────

app = FastAPI(
    title="Self-Healing RAG API",
    description="Ask questions against your knowledge base. Answers are graded and retried automatically.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# GLOBAL EXCEPTION HANDLER
# ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches any unhandled exception from any endpoint.

    WHY a global handler?
    Without it, unhandled exceptions return a raw 500 with a Python
    traceback — leaking internal details to the client.
    With it: client gets a clean generic message, server logs the full
    traceback for debugging. (Req 8.5)
    """
    logger.exception(f"Unhandled exception on {request.method} {request.url}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again later."},
    )


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Health check endpoint. (Req 8.4)

    WHY have a health check?
    Load balancers, container orchestrators (Kubernetes), and monitoring
    tools ping /health to know if the service is alive.
    Returns 200 = service is up. Anything else = something is wrong.
    """
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, req: Request):
    """
    Main endpoint — accepts a question, returns a grounded answer.

    Flow:
    1. FastAPI validates the request body via QueryRequest (Pydantic)
    2. We call query_rag() with the pre-loaded agent from app.state
    3. FastAPI validates the response via QueryResponse (Pydantic)
    4. Returns JSON

    WHY response_model=QueryResponse?
    FastAPI will serialize the return dict to match QueryResponse exactly.
    Extra fields are stripped. Missing fields raise an error at dev time.
    This guarantees the API contract is always honored. (Req 8.2)
    """
    logger.info(f"[POST /query] question='{request.question}'")

    result = query_rag(req.app.state.agent, request.question)

    logger.info(f"[POST /query] attempts={result['attempts']} answer_len={len(result['answer'])}")

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        attempts=result["attempts"],
    )
