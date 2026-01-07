from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from app.chunking.base import Chunk


@dataclass
class QueryResponse:
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]


class BaseQueryEngine(ABC):
    @abstractmethod
    async def answer(self, query: str, context_chunks: List[Chunk]) -> QueryResponse:
        pass
