

import logging
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Same model used for ingestion AND retrieval — must never differ between the two
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "knowledge_base"


def load_documents(docs_dir: str):
    
    logger.info(f"Loading documents from: {docs_dir}")
    all_docs = []

    # --- Load .txt files ---
    txt_loader = DirectoryLoader(
        docs_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,
    )
    txt_docs = txt_loader.load()
    logger.info(f"  .txt files: {len(txt_docs)} document(s) loaded")
    all_docs.extend(txt_docs)

    # --- Load .pdf files ---
    # PyPDFLoader splits each page into a separate Document automatically
    pdf_loader = DirectoryLoader(
        docs_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        silent_errors=True,
    )
    pdf_docs = pdf_loader.load()
    logger.info(f"  .pdf files: {len(pdf_docs)} page(s) loaded")
    all_docs.extend(pdf_docs)

    logger.info(f"Total documents loaded: {len(all_docs)}")
    return all_docs


def chunk_documents(docs):
    
    logger.info("Initializing embedding model for semantic chunking...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
    )

    logger.info("Chunking documents semantically...")
    chunks = chunker.split_documents(docs)
    logger.info(f"Created {len(chunks)} chunk(s) from {len(docs)} document(s)")
    return chunks


def build_vector_store(chunks, persist_dir: str):
    
    logger.info(f"Connecting to ChromaDB at: {persist_dir}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    existing_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    # Collect already-ingested source paths
    existing_data = existing_store.get()
    existing_sources = set()
    if existing_data and existing_data.get("metadatas"):
        for meta in existing_data["metadatas"]:
            if meta and "source" in meta:
                existing_sources.add(meta["source"])

    if existing_sources:
        logger.info(f"Already ingested: {existing_sources}")

    new_chunks = [c for c in chunks if c.metadata.get("source") not in existing_sources]

    if not new_chunks:
        logger.info("No new documents to ingest — all sources already in vector store.")
        return existing_store

    logger.info(f"Ingesting {len(new_chunks)} new chunk(s)...")

    for i, chunk in enumerate(new_chunks):
        chunk.metadata["chunk_index"] = i

    store = Chroma.from_documents(
        documents=new_chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )

    logger.info(f"Vector store ready. Total chunks stored: {store._collection.count()}")
    return store


def run_ingestion(docs_dir: str = "./docs", persist_dir: str = None):
    
    if persist_dir is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    logger.info("=== Starting ingestion pipeline ===")
    docs = load_documents(docs_dir)

    if not docs:
        logger.warning("No documents found. Check your docs/ directory.")
        return

    chunks = chunk_documents(docs)
    build_vector_store(chunks, persist_dir)
    logger.info("=== Ingestion complete ===")


# Only runs when executed directly: uv run python src/ingest.py
# Does NOT run when imported by rag_agent.py or tests
if __name__ == "__main__":
    run_ingestion()
