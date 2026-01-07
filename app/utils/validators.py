from pathlib import Path
from typing import Optional
from app.core.exceptions import ValidationError


def validate_file_extension(filename: str, allowed_extensions: list = None) -> None:
    if allowed_extensions is None:
        allowed_extensions = [".pdf"]
    
    file_ext = Path(filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise ValidationError(f"File extension {file_ext} not allowed. Allowed: {allowed_extensions}")


def validate_file_size(file_size: int, max_size_mb: int) -> None:
    max_size_bytes = max_size_mb * 1024 * 1024
    if file_size > max_size_bytes:
        raise ValidationError(f"File size {file_size} exceeds maximum {max_size_bytes} bytes")


def validate_document_id(document_id: str) -> None:
    if not document_id or not document_id.strip():
        raise ValidationError("Document ID cannot be empty")
    
    if len(document_id) > 255:
        raise ValidationError("Document ID too long")


def validate_query_text(query_text: str) -> None:
    if not query_text or not query_text.strip():
        raise ValidationError("Query text cannot be empty")
    
    if len(query_text) > 10000:
        raise ValidationError("Query text too long")
