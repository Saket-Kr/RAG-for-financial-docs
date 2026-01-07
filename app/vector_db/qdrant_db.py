from typing import Any, Dict, List, Optional

import numpy as np
from app.chunking.base import Chunk
from app.core.exceptions import VectorDBError
from app.vector_db.base import BaseVectorDB


class QdrantVectorDB(BaseVectorDB):
    """
    Qdrant-based vector database implementation.

    Stores vectors and payloads (metadata) in a Qdrant collection and supports
    per-document search via payload filtering on `document_id`.
    """

    def __init__(
        self,
        persist_directory: str = "./data/vector_db",
        collection_name: str = "financial_documents",
        distance_metric: str = "cosine",
        host: str = "qdrant",
        port: int = 6333,
        use_global_collection: bool = True,
    ):
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self.host = host
        self.port = port
        self.use_global_collection = use_global_collection

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qmodels
        except ImportError:
            raise VectorDBError("qdrant-client library not installed")

        self.qmodels = qmodels
        self.client = QdrantClient(host=self.host, port=self.port)

    def _get_collection_name(self, document_id: str) -> str:
        if self.use_global_collection:
            return self.collection_name
        return f"{self.collection_name}_{document_id}"

    def _ensure_collection(self, collection_name: str, vector_size: int) -> None:
        qmodels = self.qmodels

        try:
            if collection_name in [c.name for c in self.client.get_collections().collections]:
                return

            distance = (
                qmodels.Distance.COSINE
                if self.distance_metric == "cosine"
                else qmodels.Distance.EUCLID
            )

            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=distance,
                ),
            )
        except Exception as e:
            raise VectorDBError(f"Failed to create Qdrant collection: {str(e)}") from e

    def add_documents(
        self, document_id: str, chunks: List[Chunk], embeddings: np.ndarray
    ) -> None:
        qmodels = self.qmodels

        try:
            if embeddings.size == 0:
                return

            dim = embeddings.shape[1]
            collection_name = self._get_collection_name(document_id)
            self._ensure_collection(collection_name, dim)

            points: List[qmodels.PointStruct] = []
            for chunk, vector in zip(chunks, embeddings):
                payload: Dict[str, Any] = {
                    **(chunk.metadata or {}),
                    "document_id": document_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_type": (chunk.metadata or {}).get("chunk_type", "text"),
                    "page_number": (chunk.metadata or {}).get("page_number"),
                    "text": chunk.text,
                }

                points.append(
                    qmodels.PointStruct(
                        id=chunk.chunk_id,
                        vector=vector.astype(float).tolist(),
                        payload=payload,
                    )
                )

                self.client.upsert(collection_name=collection_name, points=points)
        except Exception as e:
            raise VectorDBError(f"Failed to add documents: {str(e)}") from e

    def search(
        self, document_id: str, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        qmodels = self.qmodels

        try:
            if query_embedding.size == 0:
                return []

            collection_name = self._get_collection_name(document_id)

            # Build optional filter to restrict to document_id in global collection mode
            query_filter: Optional[qmodels.Filter] = None
            if self.use_global_collection:
                query_filter = qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id),
                        )
                    ]
                )

            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding.astype(float).tolist(),
                limit=top_k,
                query_filter=query_filter,
            )

            results: List[Dict[str, Any]] = []
            for hit in search_result:
                payload = hit.payload or {}
                results.append(
                    {
                        "chunk_id": payload.get("chunk_id"),
                        "text": payload.get("text"),
                        "metadata": payload,
                        "distance": float(hit.score),
                    }
                )

            return results
        except Exception as e:
            raise VectorDBError(f"Failed to search: {str(e)}") from e

    def delete_document(self, document_id: str) -> None:
        qmodels = self.qmodels

        try:
            collection_name = self._get_collection_name(document_id)

            if self.use_global_collection:
                # Delete all points with this document_id
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=qmodels.FilterSelector(
                        filter=qmodels.Filter(
                            must=[
                                qmodels.FieldCondition(
                                    key="document_id",
                                    match=qmodels.MatchValue(value=document_id),
                                )
                            ]
                        )
                    ),
                )
            else:
                # Drop the per-document collection
                self.client.delete_collection(collection_name=collection_name)
        except Exception as e:
            raise VectorDBError(f"Failed to delete document: {str(e)}") from e

    def document_exists(self, document_id: str) -> bool:
        try:
            collection_name = self._get_collection_name(document_id)
            if self.use_global_collection:
                # Check if any point exists with this document_id
                qmodels = self.qmodels
                res = self.client.scroll(
                    collection_name=collection_name,
                    limit=1,
                    with_payload=False,
                    scroll_filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_id",
                                match=qmodels.MatchValue(value=document_id),
                            )
                        ]
                    ),
                )
                points, _ = res
                return len(points) > 0
            else:
                # Check if collection exists
                collections = self.client.get_collections().collections
                return any(c.name == collection_name for c in collections)
        except Exception:
            return False

