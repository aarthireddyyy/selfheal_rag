# ── Stage 1: base image ──────────────────────────────────────────
# WHY python:3.12-slim?
# 'slim' strips dev tools and docs — smaller image (~150MB vs ~900MB full).
# Smaller = faster to pull, less attack surface in production.
FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────
# WHY these packages?
# - build-essential: needed to compile some Python C extensions
# - curl: useful for health checks inside the container
# - git: some pip packages fetch from git at install time
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Install uv ───────────────────────────────────────────────────
# WHY uv instead of pip?
# uv is 10-100x faster than pip for dependency resolution and install.
# We use the same tool locally and in Docker for consistency.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ── Working directory ─────────────────────────────────────────────
WORKDIR /app

# ── Copy dependency files first (layer caching) ───────────────────
# WHY copy pyproject.toml before the rest of the code?
# Docker builds in layers. If only your code changes (not dependencies),
# Docker reuses the cached dependency layer — much faster rebuilds.
COPY pyproject.toml uv.lock ./

# ── Install dependencies ──────────────────────────────────────────
# --no-dev: skip test/dev dependencies in production
# --frozen: use exact versions from uv.lock, no resolution
RUN uv sync --frozen --no-dev

# ── Copy application code ─────────────────────────────────────────
COPY src/ ./src/
COPY docs/ ./docs/

# ── Pre-download the embedding model ─────────────────────────────
# WHY do this at build time?
# The first request would trigger a ~90MB model download if we don't.
# Baking it into the image means zero cold-start delay in production.
# The model is cached in /root/.cache/huggingface/
RUN uv run python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"

# ── Run ingestion at build time (optional) ────────────────────────
# WHY? So the chroma_db is baked into the image.
# Alternative: mount chroma_db as a volume (better for production).
# Uncomment the line below if you want docs pre-ingested in the image:
# RUN uv run python src/ingest.py

# ── Expose port ───────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ─────────────────────────────────────────────────
# Docker will ping /health every 30s. If it fails 3 times → container marked unhealthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start command ─────────────────────────────────────────────────
# WHY --workers 1?
# Each worker loads the embedding model into RAM (~500MB).
# Start with 1, scale up only if you have the RAM.
# WHY --host 0.0.0.0?
# Without this, uvicorn only listens on localhost INSIDE the container
# and is unreachable from outside.
CMD ["uv", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
