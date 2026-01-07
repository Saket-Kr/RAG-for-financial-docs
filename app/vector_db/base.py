from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from app.chunking.base import Chunk


class BaseVectorDB(ABC):
    @abstractmethod
    def add_documents(self, document_id: str, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        pass

    @abstractmethod
    def search(
        self,
        document_id: str,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        pass

    @abstractmethod
    def document_exists(self, document_id: str) -> bool:
        pass
