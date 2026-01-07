from typing import List, Dict, Any
import numpy as np
from app.query.base import BaseQueryEngine, QueryResponse
from app.chunking.base import Chunk
from app.query.ollama_client import OllamaClient
from app.core.exceptions import QueryError


class DirectRetrievalStrategy(BaseQueryEngine):
    async def answer(self, query: str, context_chunks: List[Chunk]) -> QueryResponse:
        if not context_chunks:
            return QueryResponse(
                answer="No relevant information found in the document.",
                sources=[],
                confidence=0.0,
                metadata={}
            )
        
        sources = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata
            }
            for chunk in context_chunks
        ]
        
        answer = "\n\n".join([chunk.text for chunk in context_chunks])
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=1.0,
            metadata={"strategy": "direct_retrieval"}
        )


class RAGStrategy(BaseQueryEngine):
    def __init__(
        self,
        ollama_client: OllamaClient,
        temperature: float = 0.0,
        max_tokens: int = 512
    ):
        self.ollama_client = ollama_client
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def answer(self, query: str, context_chunks: List[Chunk]) -> QueryResponse:
        if not context_chunks:
            return QueryResponse(
                answer="No relevant information found in the document.",
                sources=[],
                confidence=0.0,
                metadata={}
            )
        
        context = "\n\n".join([
            f"[Source {i+1}]: {chunk.text}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        prompt = f"""Based on the following context from a financial document, answer the question accurately and concisely. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            answer = await self.ollama_client.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            sources = [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata
                }
                for chunk in context_chunks
            ]
            
            confidence = min(1.0, len(context_chunks) / 5.0)
            
            return QueryResponse(
                answer=answer.strip(),
                sources=sources,
                confidence=confidence,
                metadata={"strategy": "rag", "num_sources": len(context_chunks)}
            )
        except Exception as e:
            raise QueryError(f"Failed to generate answer: {str(e)}") from e


class MultiQueryStrategy(BaseQueryEngine):
    def __init__(
        self,
        ollama_client: OllamaClient,
        temperature: float = 0.0,
        max_tokens: int = 512
    ):
        self.ollama_client = ollama_client
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def answer(self, query: str, context_chunks: List[Chunk]) -> QueryResponse:
        query_variations = await self._generate_query_variations(query)
        
        all_chunks = set(context_chunks)
        for variation in query_variations:
            pass
        
        rag_strategy = RAGStrategy(self.ollama_client, self.temperature, self.max_tokens)
        return await rag_strategy.answer(query, list(all_chunks))

    async def _generate_query_variations(self, query: str) -> List[str]:
        prompt = f"""Generate 2-3 alternative phrasings of the following question that would help find relevant information in a document:

Original question: {query}

Alternative phrasings (one per line):"""
        
        try:
            response = await self.ollama_client.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=200
            )
            variations = [line.strip() for line in response.strip().split("\n") if line.strip()]
            return variations[:3]
        except Exception:
            return [query]


class RerankingStrategy(BaseQueryEngine):
    def __init__(
        self,
        ollama_client: OllamaClient,
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_k_rerank: int = 3
    ):
        self.ollama_client = ollama_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k_rerank = top_k_rerank

    async def answer(self, query: str, context_chunks: List[Chunk]) -> QueryResponse:
        if len(context_chunks) <= self.top_k_rerank:
            rag_strategy = RAGStrategy(self.ollama_client, self.temperature, self.max_tokens)
            return await rag_strategy.answer(query, context_chunks)
        
        reranked_chunks = await self._rerank_chunks(query, context_chunks)
        top_chunks = reranked_chunks[:self.top_k_rerank]
        
        rag_strategy = RAGStrategy(self.ollama_client, self.temperature, self.max_tokens)
        return await rag_strategy.answer(query, top_chunks)

    async def _rerank_chunks(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        scores = []
        for chunk in chunks:
            prompt = f"""Rate how relevant this text is to the question on a scale of 0-10:

Question: {query}

Text: {chunk.text[:500]}

Relevance score (0-10):"""
            
            try:
                response = await self.ollama_client.generate(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=10
                )
                score = float(response.strip().split()[0]) if response.strip() else 0.0
                scores.append((score, chunk))
            except Exception:
                scores.append((0.0, chunk))
        
        scores.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scores]
