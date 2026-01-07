import uuid

import numpy as np
import pytest
from app.chunking.base import Chunk
from app.vector_db.factory import VectorDBFactory


def test_vector_db_factory(temp_dir):
    db = VectorDBFactory.create_db("faiss", persist_directory=f"{temp_dir}/vector_db")
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


def test_qdrant_add_and_search(monkeypatch):
    """
    Lightweight test that QdrantVectorDB creates points and maps results correctly.
    Uses monkeypatch to avoid requiring a running Qdrant instance.
    """
    from app.vector_db.qdrant_db import QdrantVectorDB

    class DummyPoint:
        def __init__(self, score, payload):
            self.score = score
            self.payload = payload

    class DummyClient:
        def __init__(self, *args, **kwargs):
            self.upsert_called = False

        def get_collections(self):
            class C:
                collections = []

            return C()

        def recreate_collection(self, *args, **kwargs):
            pass

        def upsert(self, collection_name, points):
            self.upsert_called = True

        def search(self, collection_name, query_vector, limit, query_filter=None):
            return [
                DummyPoint(
                    0.1,
                    {
                        "chunk_id": "chunk-1",
                        "text": "Test chunk 1",
                        "document_id": "test_doc",
                        "chunk_type": "text",
                    },
                )
            ]

        def delete(self, *args, **kwargs):
            pass

        def delete_collection(self, *args, **kwargs):
            pass

        def scroll(self, *args, **kwargs):
            return ([], None)

    # Monkeypatch QdrantClient used inside QdrantVectorDB
    import qdrant_client

    monkeypatch.setattr(qdrant_client, "QdrantClient", DummyClient)

    db = QdrantVectorDB(
        persist_directory="./data/vector_db",
        collection_name="financial_documents",
        distance_metric="cosine",
        host="qdrant",
        port=6333,
        use_global_collection=True,
    )

    document_id = "test_doc"
    chunks = [
        Chunk(text="Test chunk 1", chunk_id="chunk-1", metadata={}),
    ]

    embeddings = np.random.rand(1, 768).astype(np.float32)

    db.add_documents(document_id, chunks, embeddings)

    query_embedding = np.random.rand(768).astype(np.float32)
    results = db.search(document_id, query_embedding, top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-1"
