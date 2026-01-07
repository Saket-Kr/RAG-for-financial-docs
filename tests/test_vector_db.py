import uuid

import numpy as np
import pytest

from app.chunking.base import Chunk
from app.vector_db.factory import VectorDBFactory


def test_vector_db_factory(temp_dir):
    db = VectorDBFactory.create_db("chroma", persist_directory=f"{temp_dir}/vector_db")
    assert db is not None


def test_chroma_add_and_search(temp_dir):
    db = VectorDBFactory.create_db("chroma", persist_directory=f"{temp_dir}/vector_db")

    document_id = "test_doc"
    chunks = [
        Chunk(text="Test chunk 1", chunk_id=str(uuid.uuid4()), metadata={}),
        Chunk(text="Test chunk 2", chunk_id=str(uuid.uuid4()), metadata={}),
    ]

    embeddings = np.random.rand(2, 768).astype(np.float32)

    db.add_documents(document_id, chunks, embeddings)

    query_embedding = np.random.rand(768).astype(np.float32)
    results = db.search(document_id, query_embedding, top_k=2)

    assert len(results) <= 2
