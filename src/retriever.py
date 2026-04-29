"""
retriever.py — Hybrid Retrieval + Reranking

WHY a separate file?
The retrieve_node in rag_agent.py was doing too much.
Separating retrieval into its own module makes it:
- Testable in isolation
- Swappable (change retrieval strategy without touching the agent)
- Readable — each concern has one home

RETRIEVAL PIPELINE:
1. Vector Search   — semantic similarity via ChromaDB (finds meaning matches)
2. BM25 Search     — keyword matching (finds exact term matches)
3. Merge           — combine both result sets, deduplicate
4. Rerank          — cross-encoder scores every candidate against the query
5. Return top-k    — only the best chunks go to the LLM

WHY this order?
Vector + BM25 cast a wide net (20 candidates).
Reranker is slow but precise — it's fine to run on 20 candidates.
The LLM only sees the top-4 after reranking — best signal, minimal noise.
"""

import logging
import string
from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Cross-encoder reranker model
# WHY this model?
# ms-marco is trained specifically for passage ranking — exactly our use case.
# MiniLM-L6 is the small/fast variant: ~80MB, runs on CPU in ~50ms per candidate.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Cached reranker — same pattern as embeddings, load once reuse forever
_reranker: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info("Loading reranker model...")
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("Reranker ready.")
    return _reranker


def _tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for BM25.

    WHY lowercase + remove punctuation?
    BM25 is case-sensitive by default. "Python" and "python" would be
    treated as different words. Normalizing ensures consistent matching.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def hybrid_retrieve(
    query: str,
    store: Chroma,
    k: int = 4,
    fetch_k: int = 10,
    min_reranker_score: float = -5.0,
) -> list[str]:
    """
    Retrieves the top-k most relevant chunks using hybrid search + reranking.

    Args:
        query:              the user's question
        store:              ChromaDB vector store
        k:                  chunks to return to the LLM
        fetch_k:            candidates each method fetches before reranking
        min_reranker_score: if the best chunk scores below this, return empty
                            (signals the query has nothing to do with the KB)

    Returns empty list if no chunk is relevant enough — the agent will
    then generate "I cannot answer" which the grader fails → give_up.
    A score of -5.0 is a conservative threshold — clearly irrelevant queries
    (like injection attempts or off-topic questions) score below -8.
    """

    # ── Step 1: Vector search ─────────────────────────────────────
    logger.info(f"[hybrid_retrieve] vector search for: '{query[:80]}...'")
    vector_results = store.similarity_search(query, k=fetch_k)
    vector_texts = [doc.page_content for doc in vector_results]
    logger.info(f"[hybrid_retrieve] vector search returned {len(vector_texts)} chunks")

    # ── Step 2: BM25 search ───────────────────────────────────────
    # Load all chunks from ChromaDB to build the BM25 corpus
    all_data = store.get()
    all_texts = all_data.get("documents", [])

    if not all_texts:
        logger.warning("[hybrid_retrieve] ChromaDB returned no documents for BM25")
        bm25_texts = []
    else:
        tokenized_corpus = [_tokenize(t) for t in all_texts]
        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = _tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        # Get indices of top fetch_k scores
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:fetch_k]
        bm25_texts = [all_texts[i] for i in top_indices if scores[i] > 0]
        logger.info(f"[hybrid_retrieve] BM25 returned {len(bm25_texts)} chunks")

    # ── Step 3: Merge + deduplicate ───────────────────────────────
    seen = set()
    candidates = []
    for text in vector_texts + bm25_texts:
        # Use first 100 chars as dedup key — avoids exact string comparison on long texts
        key = text[:100]
        if key not in seen:
            seen.add(key)
            candidates.append(text)

    logger.info(f"[hybrid_retrieve] {len(candidates)} unique candidates after merge")

    if not candidates:
        return []

    # ── Step 4: Rerank ────────────────────────────────────────────
    # CrossEncoder expects list of (query, passage) pairs
    reranker = get_reranker()
    pairs = [(query, chunk) for chunk in candidates]
    scores = reranker.predict(pairs)

    # ── Step 5: Return top-k ──────────────────────────────────────
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top_score = ranked[0][0]

    # Relevance gate — if even the best chunk scores below threshold,
    # the query has nothing to do with the knowledge base
    if top_score < min_reranker_score:
        logger.warning(
            f"[hybrid_retrieve] top reranker score {top_score:.3f} below threshold "
            f"{min_reranker_score} — returning empty (query not relevant to KB)"
        )
        return []

    top_chunks = [text for _, text in ranked[:k]]

    logger.info(
        f"[hybrid_retrieve] reranked {len(candidates)} → returning top {len(top_chunks)} chunks"
        f" (top score: {top_score:.3f})"
    )

    return top_chunks
