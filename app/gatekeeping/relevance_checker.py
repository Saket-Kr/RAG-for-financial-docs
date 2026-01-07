from typing import Tuple
import numpy as np
from app.core.exceptions import GatekeepingError


class RelevanceChecker:
    def __init__(self, threshold: float = 0.3, rejection_message: str = None):
        self.threshold = threshold
        self.rejection_message = rejection_message or "This query does not appear to be related to the document."

    def check_relevance(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> Tuple[bool, float]:
        if len(document_embeddings) == 0:
            return False, 0.0
        
        try:
            similarities = np.dot(document_embeddings, query_embedding) / (
                np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            max_similarity = float(np.max(similarities))
            is_relevant = max_similarity >= self.threshold
            return is_relevant, max_similarity
        except Exception as e:
            raise GatekeepingError(f"Failed to check relevance: {str(e)}") from e

    def check_relevance_from_results(
        self,
        search_results: list
    ) -> Tuple[bool, float]:
        if not search_results:
            return False, 0.0
        
        distances = [r.get("distance", 1.0) for r in search_results if "distance" in r]
        if not distances:
            return True, 1.0
        
        min_distance = min(distances)
        
        if self.threshold < 1.0:
            similarity = 1.0 - min_distance
            is_relevant = similarity >= self.threshold
            return is_relevant, similarity
        else:
            is_relevant = min_distance <= (1.0 - self.threshold)
            return is_relevant, 1.0 - min_distance
