from typing import Dict, Type
from app.chunking.base import BaseChunker
from app.chunking.strategies import (
    FixedSizeChunker,
    SentenceChunker,
    HierarchicalChunker,
    SlidingWindowChunker,
    SemanticChunker
)
from app.core.exceptions import ValidationError


class ChunkingFactory:
    _chunkers: Dict[str, Type[BaseChunker]] = {
        "fixed": FixedSizeChunker,
        "sentence": SentenceChunker,
        "hierarchical": HierarchicalChunker,
        "sliding": SlidingWindowChunker,
        "semantic": SemanticChunker,
    }

    @classmethod
    def create_chunker(
        cls,
        strategy: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedder=None
    ) -> BaseChunker:
        if strategy not in cls._chunkers:
            raise ValidationError(f"Unknown chunking strategy: {strategy}")
        
        chunker_class = cls._chunkers[strategy]
        
        if strategy == "semantic":
            return chunker_class(chunk_size, chunk_overlap, embedder)
        else:
            return chunker_class(chunk_size, chunk_overlap)

    @classmethod
    def register_chunker(cls, name: str, chunker_class: Type[BaseChunker]) -> None:
        cls._chunkers[name] = chunker_class
