import pytest

from app.core.exceptions import EmbeddingError
from app.embeddings.factory import EmbeddingFactory


def test_embedding_factory():
    try:
        embedder = EmbeddingFactory.create_embedder("nomic_embed")
        assert embedder is not None
        assert embedder.get_dimension() == 768
    except EmbeddingError:
        pytest.skip("Nomic embed not available in test environment")
