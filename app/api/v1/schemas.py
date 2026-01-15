from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: str
    status: str
    message: str = "Document uploaded successfully"


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The question to ask about the document",
    )


class Source(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any] = {}


class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]] = None


class DocumentInfoResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]] = None
    query_count: int = 0


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int


class HealthResponse(BaseModel):
    status: str
    version: str
    services: Dict[str, str]


class ChunkResponse(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


class ChunksResponse(BaseModel):
    document_id: str
    chunk_type: Optional[str] = None
    total_chunks: int
    chunks: List[ChunkResponse]
