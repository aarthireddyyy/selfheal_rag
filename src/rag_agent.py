

import json
import logging
import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "knowledge_base"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Cache the embedding model at module load time — loads once, reused for every query
# Without this, it reloads from HuggingFace on every single retrieve call
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        # model_kwargs: run on CPU
        # encode_kwargs: normalize vectors for better cosine similarity
        # local_files_only after first download stops the HuggingFace Hub
        # version-check HTTP requests on every instantiation
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings

# The exact string returned when all retries are exhausted
# Defined as a constant so tests can assert against it reliably
GIVE_UP_MESSAGE = (
    "I don't have enough information in the knowledge base to answer this question reliably."
)


# ─────────────────────────────────────────────
# STATE SCHEMA
# ─────────────────────────────────────────────

class RAGState(TypedDict):
   
    query: str
    original_query: str
    documents: list
    answer: str
    grade: str
    grade_reason: str
    retry_count: int


# ─────────────────────────────────────────────
# NODE 1: RETRIEVE
# ─────────────────────────────────────────────

def retrieve_node(state: RAGState) -> dict:
    
    logger.info(f"[retrieve] query='{state['query']}' retry={state['retry_count']}")

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    embeddings = get_embeddings()

    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    # similarity_search returns Document objects sorted by relevance
    results = store.similarity_search(state["query"], k=4)

    # Extract just the text content — state stores strings, not Document objects
    documents = [doc.page_content for doc in results]

    logger.info(f"[retrieve] found {len(documents)} chunk(s)")
    return {"documents": documents}


# ─────────────────────────────────────────────
# NODE 2: GENERATE
# ─────────────────────────────────────────────

def generate_node(state: RAGState) -> dict:
    
    logger.info(f"[generate] retry={state['retry_count']}")

    context = "\n\n".join(state["documents"]) if state["documents"] else "No context available."

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the context does not contain enough information, say you cannot answer from the available documents.

Context:
{context}

Question: {state['query']}

Answer:"""

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,  # deterministic — we want grounded answers, not creative ones
    )

    response = llm.invoke(prompt)
    logger.info(f"[generate] answer generated ({len(response.content)} chars)")
    return {"answer": response.content}


# ─────────────────────────────────────────────
# NODE 3: GRADE
# ─────────────────────────────────────────────

def grade_node(state: RAGState) -> dict:
   
    logger.info(f"[grade] retry={state['retry_count']}")

    context = "\n\n".join(state["documents"]) if state["documents"] else "No context available."

    prompt = f"""You are a grader evaluating whether an answer is grounded in the provided context.

Context:
{context}

Answer to evaluate:
{state['answer']}

Grading rules:
1. If the answer makes specific claims that are supported by the context → grade: pass
2. If the answer says "I cannot answer", "I don't know", or "not enough information" → grade: fail
   (These responses mean retrieval failed and we should retry with a better query)
3. If the answer contains claims NOT found in the context → grade: fail

Respond with ONLY valid JSON in this exact format:
{{"grade": "pass", "reason": "brief explanation"}}
or
{{"grade": "fail", "reason": "brief explanation"}}

Do not include any text outside the JSON."""

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )

    response = llm.invoke(prompt)

    try:
        # Strip markdown code fences if the LLM wraps JSON in ```json ... ```
        raw = response.content.strip().strip("```json").strip("```").strip()
        result = json.loads(raw)
        grade = result.get("grade", "fail")
        reason = result.get("reason", "")
    except json.JSONDecodeError:
        logger.warning("[grade] Failed to parse grader JSON — defaulting to fail")
        grade = "fail"
        reason = "parse error"

    logger.info(f"[grade] grade={grade} reason='{reason}'")
    return {"grade": grade, "grade_reason": reason}


# ─────────────────────────────────────────────
# NODE 4: REWRITE
# ─────────────────────────────────────────────

def rewrite_node(state: RAGState) -> dict:
    
    logger.info(f"[rewrite] retry={state['retry_count']} → {state['retry_count'] + 1}")

    prompt = f"""You are a query rewriter helping improve document retrieval.

The original question was:
{state['original_query']}

A previous attempt to answer this question failed because:
{state['grade_reason']}

Rewrite the question using different phrasing and vocabulary while preserving the original intent.
Try a different angle that might retrieve more relevant documents.
Return ONLY the rewritten question, nothing else."""

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,  # slight creativity helps find different angles
    )

    response = llm.invoke(prompt)
    rewritten = response.content.strip()

    logger.info(f"[rewrite] original='{state['original_query']}' → rewritten='{rewritten}'")
    return {
        "query": rewritten,
        "retry_count": state["retry_count"] + 1,
    }


# ─────────────────────────────────────────────
# NODE 5: GIVE UP
# ─────────────────────────────────────────────

def give_up_node(state: RAGState) -> dict:
    
    logger.info(f"[give_up] all retries exhausted after {state['retry_count']} attempt(s)")
    return {"answer": GIVE_UP_MESSAGE}


# ─────────────────────────────────────────────
# CONDITIONAL EDGE
# ─────────────────────────────────────────────

def should_retry(state: RAGState) -> str:
    
    if state["grade"] == "pass":
        logger.info("[router] grade=pass → returning answer")
        return "end"
    elif state["retry_count"] >= 2:
        logger.info("[router] grade=fail, retries exhausted → give_up")
        return "give_up"
    else:
        logger.info(f"[router] grade=fail, retry_count={state['retry_count']} → rewrite")
        return "rewrite"


# ─────────────────────────────────────────────
# GRAPH WIRING + PUBLIC API
# ─────────────────────────────────────────────

def create_rag_agent(persist_dir: str = None):
    
    if persist_dir:
        os.environ["CHROMA_PERSIST_DIR"] = persist_dir

    graph = StateGraph(RAGState)

    # Add nodes — each is a Python function
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("give_up", give_up_node)

    # Add fixed edges (always go from A to B)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "grade")

    # Conditional edge: after grading, should_retry() decides where to go
    # The dict maps return values of should_retry() to node names
    graph.add_conditional_edges(
        "grade",
        should_retry,
        {
            "end": END,
            "rewrite": "rewrite",
            "give_up": "give_up",
        },
    )

    # After rewrite, go back to retrieve (the retry loop)
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("give_up", END)

    return graph.compile()


def query_rag(agent, question: str) -> dict:
    
    initial_state: RAGState = {
        "query": question,
        "original_query": question,
        "documents": [],
        "answer": "",
        "grade": "",
        "grade_reason": "",
        "retry_count": 0,
    }

    final_state = agent.invoke(initial_state)

    return {
        "answer": final_state["answer"],
        "sources": final_state.get("documents", []),
        "attempts": final_state.get("retry_count", 0) + 1,
    }


# ─────────────────────────────────────────────
# QUICK TEST — run directly to verify the agent
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Self-Healing RAG Agent ===")
    print("Make sure you've run ingest.py first!")
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    agent = create_rag_agent()

    while True:
        question = input("Q: ").strip()
        if not question or question.lower() in ("quit", "exit"):
            break
        result = query_rag(agent, question)
        print(f"A: {result['answer']}")
        print(f"   attempts={result['attempts']} | chunks_retrieved={len(result['sources'])}")
        print("-" * 60)
