import logging
from typing import List, Optional

import httpx
import numpy as np

from app.core.exceptions import EmbeddingError
from app.embeddings.base import BaseEmbedder

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
        batch_size: int = 32,
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
        # Remove trailing slash if present
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

            logger.debug(
                f"Making single embedding request to {self.single_embed_url} "
                f"for prompt length: {len(prompt)}"
            )

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

                logger.debug(
                    f"Successfully received embedding, dimension: {len(embedding)}"
                )
                return embedding

        except httpx.TimeoutException as e:
            logger.error(f"Timeout calling API: {str(e)}")
            raise EmbeddingError(
                f"Request timeout after {self.timeout}s while calling API"
            ) from e
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error calling API: {e.response.status_code} - {e.response.text}"
            )
            raise EmbeddingError(
                f"HTTP {e.response.status_code} error from API: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            logger.error(f"Request error calling API: {str(e)}")
            raise EmbeddingError(f"Failed to connect to API: {str(e)}") from e
        except ValueError as e:
            logger.error(f"Invalid JSON response from API: {str(e)}")
            raise EmbeddingError(f"Invalid response format from API: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error calling API: {str(e)}", exc_info=True)
            raise EmbeddingError(
                f"Unexpected error generating embedding: {str(e)}"
            ) from e

    def _make_batch_embedding_request(self, texts: List[str]) -> List[List[float]]:
        """
        Make HTTP request to batch embedding API endpoint.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (list of floats)

        Raises:
            EmbeddingError: If the API request fails or returns unexpected format
        """
        if not texts:
            return []

        try:
            headers = {"Content-Type": "application/json"}

            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {"model": self.model_name, "input": texts}

            logger.debug(
                f"Making batch embedding request to {self.batch_embed_url} "
                f"for {len(texts)} texts"
            )

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

                logger.debug(f"Successfully received {len(embeddings)} embeddings")
                return embeddings

        except httpx.TimeoutException as e:
            logger.error(f"Timeout calling API: {str(e)}")
            raise EmbeddingError(
                f"Request timeout after {self.timeout}s while calling API"
            ) from e
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error calling API: {e.response.status_code} - {e.response.text}"
            )
            raise EmbeddingError(
                f"HTTP {e.response.status_code} error from API: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            logger.error(f"Request error calling API: {str(e)}")
            raise EmbeddingError(f"Failed to connect to API: {str(e)}") from e
        except ValueError as e:
            logger.error(f"Invalid JSON response from API: {str(e)}")
            raise EmbeddingError(f"Invalid response format from API: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error calling API: {str(e)}", exc_info=True)
            raise EmbeddingError(
                f"Unexpected error generating embeddings: {str(e)}"
            ) from e

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

                logger.debug(
                    f"Processing batch {batch_num}/{total_batches} "
                    f"({len(batch)} texts)"
                )

                # Use batch embedding endpoint
                batch_embeddings = self._make_batch_embedding_request(batch)
                embeddings.extend(batch_embeddings)

            result = np.array(embeddings)
            logger.debug(
                f"Generated embeddings for {len(texts)} texts, "
                f"shape: {result.shape}"
            )
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
            logger.debug(f"Generating query embedding for query length: {len(query)}")

            # Use single embedding endpoint
            embedding_list = self._make_single_embedding_request(query)
            embedding = np.array(embedding_list)

            logger.debug(f"Generated query embedding, shape: {embedding.shape}")
            return embedding

        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {str(e)}", exc_info=True)
            raise EmbeddingError(f"Failed to generate query embedding: {str(e)}") from e

    def get_dimension(self) -> int:
        """Return the dimension of embeddings generated by this model."""
        return self._dimension
