from typing import Dict, Type
from app.query.base import BaseQueryEngine
from app.query.strategies import (
    DirectRetrievalStrategy,
    RAGStrategy,
    MultiQueryStrategy,
    RerankingStrategy
)
from app.query.ollama_client import OllamaClient
from app.core.exceptions import QueryError


class QueryEngineFactory:
    _strategies: Dict[str, Type[BaseQueryEngine]] = {
        "direct_retrieval": DirectRetrievalStrategy,
        "rag": RAGStrategy,
        "multi_query": MultiQueryStrategy,
        "reranking": RerankingStrategy,
    }

    @classmethod
    def create_engine(
        cls,
        strategy: str,
        ollama_client: OllamaClient,
        temperature: float = 0.0,
        max_tokens: int = 512
    ) -> BaseQueryEngine:
        if strategy not in cls._strategies:
            raise QueryError(f"Unknown query strategy: {strategy}")
        
        strategy_class = cls._strategies[strategy]
        
        if strategy == "direct_retrieval":
            return strategy_class()
        elif strategy == "rag":
            return strategy_class(ollama_client, temperature, max_tokens)
        elif strategy == "multi_query":
            return strategy_class(ollama_client, temperature, max_tokens)
        elif strategy == "reranking":
            return strategy_class(ollama_client, temperature, max_tokens, top_k_rerank=3)
        else:
            return strategy_class()

    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[BaseQueryEngine]) -> None:
        cls._strategies[name] = strategy_class
