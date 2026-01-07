from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    text: str
    chunk_id: str
    metadata: Dict[str, Any]
    start_index: Optional[int] = None
    end_index: Optional[int] = None


class BaseChunker(ABC):
    @abstractmethod
    def chunk(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        pass
