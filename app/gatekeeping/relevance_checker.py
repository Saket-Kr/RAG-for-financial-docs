import logging
from typing import Tuple

import numpy as np
from app.core.exceptions import GatekeepingError

logger = logging.getLogger(__name__)


class RelevanceChecker:
    def __init__(
        self,
        threshold: float = 0.3,
        rejection_message: str = None,
        distance_metric: str = "cosine",
        vector_db_type: str = "chroma",
    ):
        self.threshold = threshold
        self.rejection_message = (
            rejection_message
            or "This query does not appear to be related to the document."
        )
        self.distance_metric = distance_metric.lower()
        self.vector_db_type = vector_db_type.lower()

        # Validate distance_metric
        valid_metrics = ["cosine", "euclidean", "l2", "manhattan"]
        if self.distance_metric not in valid_metrics:
            logger.warning(
                f"Unknown distance_metric '{distance_metric}', defaulting to cosine behavior"
            )

        # Validate vector_db_type
        valid_db_types = ["chroma", "faiss", "qdrant"]
        if self.vector_db_type not in valid_db_types:
            logger.warning(
                f"Unknown vector_db_type '{vector_db_type}', defaulting to chroma behavior"
            )

    def check_relevance(
        self, query_embedding: np.ndarray, document_embeddings: np.ndarray
    ) -> Tuple[bool, float]:
        if len(document_embeddings) == 0:
            return False, 0.0

        try:
            similarities = np.dot(document_embeddings, query_embedding) / (
                np.linalg.norm(document_embeddings, axis=1)
                * np.linalg.norm(query_embedding)
            )
            max_similarity = float(np.max(similarities))
            is_relevant = max_similarity >= self.threshold
            return is_relevant, max_similarity
        except Exception as e:
            raise GatekeepingError(f"Failed to check relevance: {str(e)}") from e

    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert distance to similarity based on distance metric and vector DB type.

        Returns similarity in range [0, 1] where 1 = most similar, 0 = least similar.
        """
        if self.distance_metric == "cosine":
            if self.vector_db_type == "chroma":
                # ChromaDB: distance = 1 - cosine_similarity (range: 0 to 2)
                # similarity = 1 - distance, clamped to [0, 1]
                similarity = max(0.0, min(1.0, 1.0 - distance))
            elif self.vector_db_type == "qdrant":
                # Qdrant: returns similarity score directly (range: 0 to 1, higher is better)
                # The "distance" field is actually a similarity score, use it directly
                similarity = max(0.0, min(1.0, distance))
            elif self.vector_db_type == "faiss":
                # FAISS with IndexFlatIP: returns inner product (similarity, not distance)
                # Normalize to [0, 1] assuming normalized vectors
                # Inner product of normalized vectors = cosine similarity (range: -1 to 1)
                similarity = max(0.0, min(1.0, (distance + 1.0) / 2.0))
            else:
                # Default: assume distance = 1 - similarity
                similarity = max(0.0, min(1.0, 1.0 - distance))

        elif self.distance_metric in ["euclidean", "l2"]:
            if self.vector_db_type == "faiss":
                # FAISS L2: returns actual Euclidean distance (0 to infinity)
                # Use exponential decay: similarity = exp(-distance)
                similarity = float(np.exp(-distance))
            elif self.vector_db_type == "qdrant":
                # Qdrant Euclidean: returns distance (0 to infinity)
                similarity = float(np.exp(-distance))
            else:
                # Default: exponential decay
                similarity = float(np.exp(-distance))

        elif self.distance_metric == "manhattan":
            # Manhattan distance: use exponential decay
            similarity = float(np.exp(-distance))

        else:
            # Unknown metric: assume it's already a similarity or use default conversion
            similarity = max(0.0, min(1.0, 1.0 - distance))

        logger.debug(
            f"Converted distance {distance:.4f} to similarity {similarity:.4f} "
            f"(metric={self.distance_metric}, db={self.vector_db_type})"
        )
        return similarity

    def check_relevance_from_results(self, search_results: list) -> Tuple[bool, float]:
        """
        Check relevance based on search results from vector DB.
        Converts distances to similarities based on configured distance metric.
        """
        if not search_results:
            return False, 0.0

        distances = [r.get("distance", 1.0) for r in search_results if "distance" in r]
        if not distances:
            return True, 1.0

        # For distance metrics, lower is better (find minimum)
        # For similarity metrics (FAISS inner product, Qdrant cosine), higher is better (find maximum)
        if self.distance_metric == "cosine" and self.vector_db_type == "faiss":
            # FAISS IndexFlatIP returns similarity (higher is better)
            max_distance = max(distances)
            similarity = self._distance_to_similarity(max_distance)
        elif self.distance_metric == "cosine" and self.vector_db_type == "qdrant":
            # Qdrant cosine returns similarity score (higher is better)
            max_distance = max(distances)
            similarity = self._distance_to_similarity(max_distance)
        else:
            # All other cases: distance (lower is better)
            min_distance = min(distances)
            similarity = self._distance_to_similarity(min_distance)

        is_relevant = similarity >= self.threshold
        return is_relevant, similarity
