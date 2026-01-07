import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.database import Database, Document, Query
from app.core.exceptions import DocumentNotFoundError


class MetadataService:
    def __init__(self, db: Database):
        self.db = db

    def create_document(
        self, filename: str, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        document_id = str(uuid.uuid4())
        document = Document(
            id=document_id,
            filename=filename,
            file_path=file_path,
            status="pending",
            metadata_json=metadata,
        )

        session = self.db.get_session()
        try:
            session.add(document)
            session.commit()
            session.refresh(document)
            return document
        finally:
            session.close()

    def get_document(self, document_id: str) -> Document:
        session = self.db.get_session()
        try:
            document = (
                session.query(Document).filter(Document.id == document_id).first()
            )
            if not document:
                raise DocumentNotFoundError(f"Document {document_id} not found")
            return document
        finally:
            session.close()

    def update_document_status(self, document_id: str, status: str) -> None:
        session = self.db.get_session()
        try:
            document = (
                session.query(Document).filter(Document.id == document_id).first()
            )
            if not document:
                raise DocumentNotFoundError(f"Document {document_id} not found")
            document.status = status
            document.updated_at = datetime.utcnow()
            session.commit()
        finally:
            session.close()

    def update_document_metadata(
        self, document_id: str, metadata: Dict[str, Any]
    ) -> None:
        session = self.db.get_session()
        try:
            document = (
                session.query(Document).filter(Document.id == document_id).first()
            )
            if not document:
                raise DocumentNotFoundError(f"Document {document_id} not found")
            document.metadata_json = metadata
            document.updated_at = datetime.utcnow()
            session.commit()
        finally:
            session.close()

    def delete_document(self, document_id: str) -> None:
        session = self.db.get_session()
        try:
            document = (
                session.query(Document).filter(Document.id == document_id).first()
            )
            if not document:
                raise DocumentNotFoundError(f"Document {document_id} not found")
            session.delete(document)
            session.commit()
        finally:
            session.close()

    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Document]:
        session = self.db.get_session()
        try:
            return session.query(Document).offset(offset).limit(limit).all()
        finally:
            session.close()

    def create_query(
        self,
        document_id: str,
        query_text: str,
        answer: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> Query:
        query = Query(
            document_id=document_id,
            query_text=query_text,
            answer=answer,
            confidence=confidence,
        )

        session = self.db.get_session()
        try:
            session.add(query)
            session.commit()
            session.refresh(query)
            return query
        finally:
            session.close()

    def get_document_queries(self, document_id: str, limit: int = 100) -> List[Query]:
        session = self.db.get_session()
        try:
            return (
                session.query(Query)
                .filter(Query.document_id == document_id)
                .order_by(Query.created_at.desc())
                .limit(limit)
                .all()
            )
        finally:
            session.close()
