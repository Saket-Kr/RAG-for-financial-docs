from typing import Dict, Type

from app.core.config import Settings
from app.core.exceptions import EmbeddingError
from app.embeddings.base import BaseEmbedder
from app.embeddings.nomic_embedder import NomicEmbedder


class EmbeddingFactory:
    _embedders: Dict[str, Type[BaseEmbedder]] = {
        "nomic_embed": NomicEmbedder,
    }

    @classmethod
    def create_embedder(cls, embedder_type: str, settings: Settings) -> BaseEmbedder:
        if embedder_type not in cls._embedders:
            raise EmbeddingError(f"Unknown embedder type: {embedder_type}")

        embedder_class = cls._embedders[embedder_type]

        if embedder_type == "nomic_embed":
            return embedder_class(
                settings.embeddings.model_name,
                settings.embeddings.device,
                settings.embeddings.batch_size,
                settings.embeddings.api_url,
                settings.embeddings.api_key,
            )
        else:
            return embedder_class()

    @classmethod
    def register_embedder(cls, name: str, embedder_class: Type[BaseEmbedder]) -> None:
        cls._embedders[name] = embedder_class
