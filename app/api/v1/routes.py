import logging
from typing import Optional

from app.api.dependencies import (
    get_document_service,
    get_metadata_service,
    get_settings,
)
from app.api.v1.schemas import (
    ChunkResponse,
    ChunksResponse,
    DocumentInfoResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    Source,
)
from app.core.config import Settings
from app.core.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingError,
    GatekeepingError,
    ValidationError,
)
from app.services.document_service import DocumentService
from app.services.metadata_service import MetadataService
from app.utils.validators import (
    validate_file_extension,
    validate_file_size,
    validate_query_text,
)
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
    settings: Settings = Depends(get_settings),
):
    try:
        validate_file_extension(file.filename)

        file_content = await file.read()
        validate_file_size(len(file_content), settings.storage.max_file_size_mb)

        document_id = await document_service.process_document(
            file_content, file.filename
        )

        return DocumentUploadResponse(document_id=document_id, status="processing")
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload document",
        )


@router.post("/documents/{document_id}/query", response_model=QueryResponse)
async def query_document(
    document_id: str,
    query_request: QueryRequest,
    document_service: DocumentService = Depends(get_document_service),
):
    try:
        validate_query_text(query_request.query)

        result = await document_service.query_document(document_id, query_request.query)

        sources = [
            Source(
                chunk_id=src["chunk_id"],
                text=src["text"],
                metadata=src.get("metadata", {}),
            )
            for src in result["sources"]
        ]

        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            confidence=result["confidence"],
            metadata=result.get("metadata", {}),
        )
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )
    except GatekeepingError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error querying document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query",
        )


@router.get("/documents/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: str, metadata_service: MetadataService = Depends(get_metadata_service)
):
    try:
        document = metadata_service.get_document(document_id)
        return DocumentStatusResponse(
            document_id=document.id,
            filename=document.filename,
            status=document.status,
            created_at=document.created_at,
            updated_at=document.updated_at,
            metadata=document.metadata_json,
        )
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )


@router.get("/documents/{document_id}", response_model=DocumentInfoResponse)
async def get_document_info(
    document_id: str, metadata_service: MetadataService = Depends(get_metadata_service)
):
    try:
        document = metadata_service.get_document(document_id)
        queries = metadata_service.get_document_queries(document_id, limit=1)

        return DocumentInfoResponse(
            document_id=document.id,
            filename=document.filename,
            status=document.status,
            created_at=document.created_at,
            updated_at=document.updated_at,
            metadata=document.metadata_json,
            query_count=len(queries),
        )
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )


@router.get("/documents/{document_id}/chunks", response_model=ChunksResponse)
async def get_document_chunks(
    document_id: str,
    chunk_type: Optional[str] = None,
    document_service: DocumentService = Depends(get_document_service),
):
    """
    Retrieve all chunks for a document.

    Args:
        document_id: ID of the document
        chunk_type: Optional filter - "text" or "table". If blank or invalid, returns all chunks.
    """
    try:
        chunks = document_service.get_all_chunks(document_id, chunk_type)

        chunk_responses = [
            ChunkResponse(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                metadata=chunk.get("metadata", {}),
            )
            for chunk in chunks
        ]

        # Determine the effective chunk_type for response
        effective_chunk_type = None
        valid_chunk_types = {"text", "table"}
        if chunk_type and chunk_type.lower() in valid_chunk_types:
            effective_chunk_type = chunk_type.lower()

        return ChunksResponse(
            document_id=document_id,
            chunk_type=effective_chunk_type,
            total_chunks=len(chunk_responses),
            chunks=chunk_responses,
        )
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )
    except Exception as e:
        logger.error(f"Error retrieving chunks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chunks",
        )


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str, document_service: DocumentService = Depends(get_document_service)
):
    try:
        await document_service.delete_document(document_id)
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document",
        )


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)):
    from app.query.ollama_client import OllamaClient

    ollama_client = OllamaClient(
        settings.query_answering.llm.base_url,
        settings.query_answering.llm.model,
        settings.query_answering.llm.timeout,
    )

    ollama_healthy = await ollama_client.health_check()

    services = {
        "database": "healthy",
        "ollama": "healthy" if ollama_healthy else "unhealthy",
    }

    overall_status = (
        "healthy" if all(v == "healthy" for v in services.values()) else "degraded"
    )

    return HealthResponse(
        status=overall_status, version=settings.app.version, services=services
    )
