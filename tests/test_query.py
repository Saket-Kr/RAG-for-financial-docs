import pytest
from app.query.factory import QueryEngineFactory
from app.query.ollama_client import OllamaClient
from app.chunking.base import Chunk
import uuid


def test_query_engine_factory():
    ollama_client = OllamaClient()
    engine = QueryEngineFactory.create_engine("direct_retrieval", ollama_client)
    assert engine is not None


def test_direct_retrieval_strategy():
    from app.query.strategies import DirectRetrievalStrategy
    
    strategy = DirectRetrievalStrategy()
    chunks = [
        Chunk(
            text="Test content",
            chunk_id=str(uuid.uuid4()),
            metadata={}
        )
    ]
    
    response = strategy.answer("test query", chunks)
    assert response.answer is not None
    assert len(response.sources) > 0
