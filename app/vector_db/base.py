from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from app.chunking.base import Chunk


class BaseVectorDB(ABC):
    @abstractmethod
    def add_documents(
        self, document_id: str, chunks: List[Chunk], embeddings: np.ndarray
    ) -> None:
        pass

    @abstractmethod
    def search(
        self, document_id: str, query_embedding: np.ndarray, top_k: int = 5, chunk_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        pass

    @abstractmethod
    def document_exists(self, document_id: str) -> bool:
        pass

    @abstractmethod
    def get_all_chunks(
        self, document_id: str, chunk_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a document, optionally filtered by chunk_type.

        Args:
            document_id: ID of the document
            chunk_type: Optional filter for chunk type ("text" or "table").
                       If None, empty, or invalid, returns all chunks.

        Returns:
            List of chunk dictionaries with chunk_id, text, and metadata
        """
        pass
