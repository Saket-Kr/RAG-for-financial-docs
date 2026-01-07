import logging
from typing import Any, Dict, Optional

import httpx

from app.core.exceptions import LLMError

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://ollama:11434",
        model: str = "mistral:7b-instruct",
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def generate(
        self, prompt: str, temperature: float = 0.0, max_tokens: int = 512, **kwargs
    ) -> str:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                            **kwargs,
                        },
                    },
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")
        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling Ollama: {str(e)}")
            raise LLMError(f"Failed to call Ollama API: {str(e)}") from e
        except Exception as e:
            logger.error(f"Error calling Ollama: {str(e)}")
            raise LLMError(f"Unexpected error calling Ollama: {str(e)}") from e

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
