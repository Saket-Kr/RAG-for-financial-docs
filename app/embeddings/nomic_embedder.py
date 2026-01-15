import logging
from typing import List, Optional

import httpx
import numpy as np
from app.core.exceptions import EmbeddingError
from app.embeddings.base import BaseEmbedder
from app.utils.retry import retry_http

logger = logging.getLogger(__name__)


class NomicEmbedder(BaseEmbedder):
    """
    Nomic embedding generator using HTTP API calls.

    Uses API endpoints:
    - Single embedding: /api/embeddings with 'prompt'
    - Batch embeddings: /api/embed with 'input' array
    """

    DEFAULT_TIMEOUT = 30.0

    def __init__(
        self,
        model_name: str = "nomic-embed-text:latest",
        device: str = "cpu",
        batch_size: int = 8,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """
        Initialize Nomic embedder with HTTP API configuration.

        Args:
            model_name: Name of the nomic embedding model to use
            device: Device specification (kept for compatibility, not used in HTTP mode)
            batch_size: Number of texts to process in each batch
            api_url: Base API URL (defaults to localhost:11434). Will derive endpoints from this.
            api_key: API key for authentication (optional)
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._dimension = 768
        base_url = api_url
        base_url = base_url.rstrip("/")
        self.base_url = base_url
        self.single_embed_url = f"{base_url}/api/embeddings"
        self.batch_embed_url = f"{base_url}/api/embed"
        self.api_key = api_key
        self.timeout = timeout

        logger.info(
            f"Initialized NomicEmbedder with model={model_name}, "
            f"base_url={self.base_url}, batch_size={batch_size}"
        )

    @retry_http(max_retries=3, initial_backoff=1.0)
    def _make_single_embedding_request(self, prompt: str) -> List[float]:
        """
        Make HTTP request to single embedding API endpoint.

        Args:
            prompt: Single text string to embed

        Returns:
            Embedding vector (list of floats)

        Raises:
            EmbeddingError: If the API request fails or returns unexpected format
        """
        try:
            headers = {"Content-Type": "application/json"}

            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {"model": self.model_name, "prompt": prompt}

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.single_embed_url, headers=headers, json=payload
                )
                response.raise_for_status()
                result = response.json()

                # returns embedding in 'embedding' key
                if "embedding" in result:
                    embedding = result["embedding"]
                elif (
                    "data" in result
                    and isinstance(result["data"], list)
                    and len(result["data"]) > 0
                ):
                    # OpenAI-compatible format
                    embedding = result["data"][0].get("embedding", [])
                else:
                    logger.error(
                        f"Unexpected API response format: {list(result.keys())}"
                    )
                    raise EmbeddingError(
                        f"Unexpected API response format. "
                        f"Expected 'embedding' or 'data' key, got: {list(result.keys())}"
                    )

                # Validate embedding format
                if not isinstance(embedding, list) or not all(
                    isinstance(x, (int, float)) for x in embedding
                ):
                    raise EmbeddingError(
                        f"Invalid embedding format: expected list of numbers"
                    )

                return embedding

        except httpx.TimeoutException as e:
            msg = (
                f"Embedding API timeout "
                f"[op=single, url={self.single_embed_url}, model={self.model_name}, "
                f"timeout={self.timeout}s]: {e}"
            )
            logger.error(msg, exc_info=True)
            raise EmbeddingError(msg) from e
        except httpx.HTTPStatusError as e:
            msg = (
                f"Embedding API HTTP error "
                f"[op=single, url={self.single_embed_url}, model={self.model_name}, "
                f"status={e.response.status_code}]: {e.response.text}"
            )
            logger.error(msg, exc_info=True)
            raise EmbeddingError(msg) from e
        except httpx.RequestError as e:
            msg = (
                f"Embedding API connection error "
                f"[op=single, url={self.single_embed_url}, model={self.model_name}]: {e}"
            )
            logger.error(msg, exc_info=True)
            raise EmbeddingError(msg) from e
        except ValueError as e:
            msg = (
                f"Embedding API invalid response "
                f"[op=single, url={self.single_embed_url}, model={self.model_name}]: {e}"
            )
            logger.error(msg, exc_info=True)
            raise EmbeddingError(msg) from e
        except Exception as e:
            msg = (
                f"Embedding API unexpected error "
                f"[op=single, url={self.single_embed_url}, model={self.model_name}]: {e}"
            )
            logger.error(msg, exc_info=True)
            raise EmbeddingError(msg) from e

    @retry_http(max_retries=3, initial_backoff=1.0)
    def _make_batch_embedding_request(self, texts: List[str]) -> List[List[float]]:
        """
        Make HTTP request to batch embedding API endpoint with automatic batch splitting.

        If a batch fails with HTTP 500 (Ollama panic due to batch too large),
        automatically splits the batch in half and retries recursively.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (list of floats)

        Raises:
            EmbeddingError: If the API request fails or returns unexpected format
        """
        if not texts:
            return []

        # Try the batch as-is with retries for transient errors
        try:
            return self._make_single_batch_request(texts)
        except EmbeddingError as e:
            # Check if this is a batch size issue (HTTP 500 from Ollama panic)
            # and we can split the batch further
            error_msg = str(e).lower()
            is_batch_size_error = (
                "status=500" in error_msg
                or "status 500" in error_msg
                or 'status_code": 500' in error_msg
            )
            can_split = len(texts) > 1

            if is_batch_size_error and can_split:
                # Split batch and retry recursively
                logger.warning(
                    f"Batch of {len(texts)} texts too large, splitting in half and retrying"
                )
                mid = len(texts) // 2
                left_embeddings = self._make_batch_embedding_request(texts[:mid])
                right_embeddings = self._make_batch_embedding_request(texts[mid:])
                return left_embeddings + right_embeddings

            # Can't split further or not a batch size error - re-raise
            raise

    def _make_single_batch_request(self, texts: List[str]) -> List[List[float]]:
        """
        Internal method that makes the actual HTTP request to the batch API.

        This method contains the HTTP request logic and is called by
        _make_batch_embedding_request which handles retries and splitting.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (list of floats)

        Raises:
            EmbeddingError: If the API request fails or returns unexpected format
        """
        headers = {"Content-Type": "application/json"}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {"model": self.model_name, "input": texts}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.batch_embed_url, headers=headers, json=payload
                )
                response.raise_for_status()
                result = response.json()

                # returns embeddings in 'embeddings' key
                if "embeddings" in result:
                    embeddings = result["embeddings"]
                elif "data" in result and isinstance(result["data"], list):
                    # OpenAI-compatible format
                    embeddings = [item.get("embedding", []) for item in result["data"]]
                else:
                    logger.error(
                        f"Unexpected API response format: {list(result.keys())}"
                    )
                    raise EmbeddingError(
                        f"Unexpected API response format. "
                        f"Expected 'embeddings' or 'data' key, got: {list(result.keys())}"
                    )

                # Validate embeddings format
                if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                    raise EmbeddingError(
                        f"Invalid embeddings response: expected {len(texts)} embeddings, "
                        f"got {len(embeddings) if isinstance(embeddings, list) else 'non-list'}"
                    )

                # Validate each embedding is a list of numbers
                for i, emb in enumerate(embeddings):
                    if not isinstance(emb, list) or not all(
                        isinstance(x, (int, float)) for x in emb
                    ):
                        raise EmbeddingError(
                            f"Invalid embedding format at index {i}: expected list of numbers"
                        )

                return embeddings

        except httpx.TimeoutException as e:
            msg = (
                f"Embedding API timeout "
                f"[op=batch, url={self.batch_embed_url}, model={self.model_name}, "
                f"batch_size={len(texts)}, timeout={self.timeout}s]: {e}"
            )
            logger.error(msg, exc_info=True)
            raise EmbeddingError(msg) from e
        except httpx.HTTPStatusError as e:
            msg = (
                f"Embedding API HTTP error "
                f"[op=batch, url={self.batch_embed_url}, model={self.model_name}, "
                f"batch_size={len(texts)}, status={e.response.status_code}]: {e.response.text}"
            )
            logger.error(msg, exc_info=True)
            raise EmbeddingError(msg) from e
        except httpx.RequestError as e:
            msg = (
                f"Embedding API connection error "
                f"[op=batch, url={self.batch_embed_url}, model={self.model_name}, "
                f"batch_size={len(texts)}]: {e}"
            )
            logger.error(msg, exc_info=True)
            raise EmbeddingError(msg) from e
        except ValueError as e:
            msg = (
                f"Embedding API invalid response "
                f"[op=batch, url={self.batch_embed_url}, model={self.model_name}, "
                f"batch_size={len(texts)}]: {e}"
            )
            logger.error(msg, exc_info=True)
            raise EmbeddingError(msg) from e
        except Exception as e:
            msg = (
                f"Embedding API unexpected error "
                f"[op=batch, url={self.batch_embed_url}, model={self.model_name}, "
                f"batch_size={len(texts)}]: {e}"
            )
            logger.error(msg, exc_info=True)
            raise EmbeddingError(msg) from e

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts using batch API.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape (num_texts, embedding_dimension)
        """
        if not texts:
            return np.array([])

        try:
            embeddings = []
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                batch_num = (i // self.batch_size) + 1

                logger.info(
                    f"Processing batch {batch_num}/{total_batches} "
                    f"({len(batch)} texts)"
                )

                # Use batch embedding endpoint
                batch_embeddings = self._make_batch_embedding_request(batch)
                embeddings.extend(batch_embeddings)

            result = np.array(embeddings)
            return result

        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}", exc_info=True)
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}") from e

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query text using single embedding API.

        Args:
            query: Query text string to embed

        Returns:
            numpy array of shape (embedding_dimension,)
        """
        if not query or not query.strip():
            raise EmbeddingError("Query text cannot be empty")

        try:
            # Use single embedding endpoint
            embedding_list = self._make_single_embedding_request(query)
            embedding = np.array(embedding_list)
            return embedding

        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {str(e)}", exc_info=True)
            raise EmbeddingError(f"Failed to generate query embedding: {str(e)}") from e

    def get_dimension(self) -> int:
        """Return the dimension of embeddings generated by this model."""
        return self._dimension
