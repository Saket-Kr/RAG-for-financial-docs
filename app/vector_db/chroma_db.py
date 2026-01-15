from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from app.chunking.base import Chunk
from app.core.exceptions import VectorDBError
from app.vector_db.base import BaseVectorDB


class ChromaVectorDB(BaseVectorDB):
    def __init__(
        self,
        persist_directory: str = "./data/vector_db",
        collection_name: str = "financial_documents",
        distance_metric: str = "cosine",
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self._collections: Dict[str, Any] = {}

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        except ImportError:
            raise VectorDBError("chromadb library not installed")

    def _get_collection(self, document_id: str):
        collection_id = f"{self.collection_name}_{document_id}"

        if collection_id not in self._collections:
            try:
                self._collections[collection_id] = self.client.get_or_create_collection(
                    name=collection_id, metadata={"document_id": document_id}
                )
            except Exception as e:
                raise VectorDBError(f"Failed to get/create collection: {str(e)}") from e

        return self._collections[collection_id]

    def add_documents(
        self, document_id: str, chunks: List[Chunk], embeddings: np.ndarray
    ) -> None:
        try:
            collection = self._get_collection(document_id)

            ids = [chunk.chunk_id for chunk in chunks]
            texts = [chunk.text for chunk in chunks]
            metadatas = [
                {
                    **chunk.metadata,
                    "document_id": document_id,
                    "chunk_id": chunk.chunk_id,
                }
                for chunk in chunks
            ]

            collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
            )
        except Exception as e:
            raise VectorDBError(f"Failed to add documents: {str(e)}") from e

    def search(
        self, document_id: str, query_embedding: np.ndarray, top_k: int = 5, chunk_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        try:
            collection = self._get_collection(document_id)

            results = collection.query(
                query_embeddings=[query_embedding.tolist()], n_results=top_k
            )

            search_results = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    search_results.append(
                        {
                            "chunk_id": results["ids"][0][i],
                            "text": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "distance": (
                                results["distances"][0][i]
                                if "distances" in results
                                else None
                            ),
                        }
                    )

            return search_results
        except Exception as e:
            raise VectorDBError(f"Failed to search: {str(e)}") from e

    def delete_document(self, document_id: str) -> None:
        try:
            collection_id = f"{self.collection_name}_{document_id}"
            if collection_id in self._collections:
                del self._collections[collection_id]

            try:
                self.client.delete_collection(name=collection_id)
            except Exception:
                pass
        except Exception as e:
            raise VectorDBError(f"Failed to delete document: {str(e)}") from e

    def document_exists(self, document_id: str) -> bool:
        try:
            collection_id = f"{self.collection_name}_{document_id}"
            try:
                self.client.get_collection(name=collection_id)
                return True
            except Exception:
                return False
        except Exception:
            return False

    def get_all_chunks(
        self, document_id: str, chunk_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Not implemented for ChromaDB - Qdrant only for now."""
        raise NotImplementedError("get_all_chunks is only implemented for Qdrant")
