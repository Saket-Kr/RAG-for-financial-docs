from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        pass
