import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from app.chunking.base import Chunk
from app.core.exceptions import VectorDBError
from app.vector_db.base import BaseVectorDB


class FAISSVectorDB(BaseVectorDB):
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
        self._indices: Dict[str, Any] = {}
        self._chunks_store: Dict[str, List[Chunk]] = {}

        try:
            import faiss

            self.faiss = faiss
        except ImportError:
            raise VectorDBError("faiss-cpu library not installed")

    def _get_index_path(self, document_id: str) -> Path:
        return self.persist_directory / f"{self.collection_name}_{document_id}.index"

    def _get_chunks_path(self, document_id: str) -> Path:
        return self.persist_directory / f"{self.collection_name}_{document_id}.chunks"

    def _load_index(self, document_id: str, dimension: int):
        index_path = self._get_index_path(document_id)

        if index_path.exists():
            index = self.faiss.read_index(str(index_path))
        else:
            if self.distance_metric == "cosine":
                index = self.faiss.IndexFlatIP(dimension)
            else:
                index = self.faiss.IndexFlatL2(dimension)

        return index

    def _save_index(self, document_id: str, index):
        index_path = self._get_index_path(document_id)
        self.faiss.write_index(index, str(index_path))

    def _load_chunks(self, document_id: str) -> List[Chunk]:
        chunks_path = self._get_chunks_path(document_id)
        if chunks_path.exists():
            with open(chunks_path, "rb") as f:
                return pickle.load(f)
        return []

    def _save_chunks(self, document_id: str, chunks: List[Chunk]):
        chunks_path = self._get_chunks_path(document_id)
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)

    def add_documents(
        self, document_id: str, chunks: List[Chunk], embeddings: np.ndarray
    ) -> None:
        try:
            dimension = embeddings.shape[1]
            index = self._load_index(document_id, dimension)

            if self.distance_metric == "cosine":
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                normalized_embeddings = embeddings / norms
            else:
                normalized_embeddings = embeddings.astype("float32")

            index.add(normalized_embeddings)

            existing_chunks = self._load_chunks(document_id)
            existing_chunks.extend(chunks)

            self._save_index(document_id, index)
            self._save_chunks(document_id, existing_chunks)
            self._chunks_store[document_id] = existing_chunks
        except Exception as e:
            raise VectorDBError(f"Failed to add documents: {str(e)}") from e

    def search(
        self, document_id: str, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        try:
            dimension = query_embedding.shape[0]
            index = self._load_index(document_id, dimension)

            if self.distance_metric == "cosine":
                norm = np.linalg.norm(query_embedding)
                normalized_query = (
                    (query_embedding / norm).reshape(1, -1).astype("float32")
                )
            else:
                normalized_query = query_embedding.reshape(1, -1).astype("float32")

            distances, indices = index.search(normalized_query, top_k)

            chunks = self._load_chunks(document_id)

            search_results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(chunks):
                    chunk = chunks[idx]
                    search_results.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "text": chunk.text,
                            "metadata": chunk.metadata,
                            "distance": float(distances[0][i]),
                        }
                    )

            return search_results
        except Exception as e:
            raise VectorDBError(f"Failed to search: {str(e)}") from e

    def delete_document(self, document_id: str) -> None:
        try:
            index_path = self._get_index_path(document_id)
            chunks_path = self._get_chunks_path(document_id)

            if document_id in self._indices:
                del self._indices[document_id]
            if document_id in self._chunks_store:
                del self._chunks_store[document_id]

            if index_path.exists():
                index_path.unlink()
            if chunks_path.exists():
                chunks_path.unlink()
        except Exception as e:
            raise VectorDBError(f"Failed to delete document: {str(e)}") from e

    def document_exists(self, document_id: str) -> bool:
        index_path = self._get_index_path(document_id)
        return index_path.exists()
