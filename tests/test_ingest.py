"""
Property 6: Ingestion deduplication
Validates: Requirements 1.5

Ingesting the same documents twice must not create duplicate chunks.
"""

import os
import tempfile

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.documents import Document

from src.ingest import build_vector_store, COLLECTION_NAME, EMBEDDING_MODEL
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def make_chunks(texts: list[str]) -> list[Document]:
    return [Document(page_content=t, metadata={"source": "test_doc.txt"}) for t in texts]


# Feature: self-healing-rag, Property 6: Ingestion deduplication
@settings(max_examples=10)
@given(st.lists(st.text(min_size=20, max_size=200), min_size=1, max_size=5))
def test_deduplication_property(texts):
    chunks = make_chunks(texts)
    with tempfile.TemporaryDirectory() as tmp_dir:
        build_vector_store(chunks, tmp_dir)
        count_after_first = _get_count(tmp_dir)

        build_vector_store(chunks, tmp_dir)
        count_after_second = _get_count(tmp_dir)

        assert count_after_first == count_after_second


def _get_count(persist_dir: str) -> int:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    return store._collection.count()
