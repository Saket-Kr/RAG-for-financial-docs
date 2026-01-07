import pytest

from app.chunking.factory import ChunkingFactory
from app.chunking.strategies import FixedSizeChunker


def test_chunking_factory():
    chunker = ChunkingFactory.create_chunker("fixed", chunk_size=100, chunk_overlap=10)
    assert chunker is not None


def test_fixed_size_chunker(sample_text):
    chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0
    assert all(len(chunk.text) <= 50 for chunk in chunks)
