import asyncio
import logging
from typing import Any, Dict, List, Tuple

from app.chunking.factory import ChunkingFactory
from app.chunking.table_chunker import TableChunker
from app.core.config import Settings
from app.core.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingError,
    GatekeepingError,
)
from app.embeddings.factory import EmbeddingFactory
from app.gatekeeping.relevance_checker import RelevanceChecker
from app.parsers.base import ParsedDocument, Table
from app.parsers.factory import ParserFactory
from app.query.factory import QueryEngineFactory
from app.query.ollama_client import OllamaClient
from app.services.metadata_service import MetadataService
from app.services.storage_service import LocalStorage
from app.vector_db.factory import VectorDBFactory

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self,
        settings: Settings,
        metadata_service: MetadataService,
        storage: LocalStorage,
    ):
        self.settings = settings
        self.metadata_service = metadata_service
        self.storage = storage

        self.parser = ParserFactory.create_parser(settings.pdf_parser.type)
        self.embedder = EmbeddingFactory.create_embedder(
            settings.embeddings.type,
            settings.embeddings.model_name,
            settings.embeddings.device,
            settings.embeddings.batch_size,
        )
        self.vector_db = VectorDBFactory.create_db(
            settings.vector_db.type,
            settings.vector_db.persist_directory,
            settings.vector_db.collection_name,
            settings.vector_db.distance_metric,
        )

        ollama_client = OllamaClient(
            settings.query_answering.llm.base_url,
            settings.query_answering.llm.model,
            settings.query_answering.llm.timeout,
        )
        self.query_engine = QueryEngineFactory.create_engine(
            settings.query_answering.strategy,
            ollama_client,
            settings.query_answering.llm.temperature,
            settings.query_answering.llm.max_tokens,
        )

        self.relevance_checker = (
            RelevanceChecker(
                settings.gatekeeping.threshold, settings.gatekeeping.rejection_message
            )
            if settings.gatekeeping.enabled
            else None
        )

        chunker_embedder = (
            self.embedder if settings.chunking.strategy == "semantic" else None
        )
        self.chunker = ChunkingFactory.create_chunker(
            settings.chunking.strategy,
            settings.chunking.chunk_size,
            settings.chunking.chunk_overlap,
            chunker_embedder,
        )

    async def process_document(self, file_content: bytes, filename: str) -> str:
        try:
            file_path = self.storage.save_file(file_content, filename)

            document = self.metadata_service.create_document(
                filename=filename,
                file_path=file_path,
                metadata={"original_filename": filename},
            )

            self.metadata_service.update_document_status(document.id, "processing")

            try:
                parsed_doc = self.parser.parse(file_path)

                all_chunks = []

                text_chunks = self.chunker.chunk(
                    parsed_doc.text,
                    metadata={
                        "document_id": document.id,
                        "filename": filename,
                        "chunk_type": "text",
                    },
                )
                all_chunks.extend(text_chunks)

                if parsed_doc.tables:
                    table_chunker = TableChunker(
                        context_window=self.settings.chunking.chunk_size
                    )

                    for table_idx, table in enumerate(parsed_doc.tables):
                        context_before, context_after, section_title, page_number = (
                            self._extract_table_context(parsed_doc, table, table_idx)
                        )

                        table_chunk = table_chunker.chunk_table(
                            table=table,
                            context_before=context_before,
                            context_after=context_after,
                            table_index=table_idx,
                            section_title=section_title,
                            page_number=page_number,
                        )
                        all_chunks.append(table_chunk)

                if not all_chunks:
                    raise DocumentProcessingError("No chunks generated from document")

                chunk_texts = [chunk.text for chunk in all_chunks]
                embeddings = self.embedder.embed_texts(chunk_texts)

                self.vector_db.add_documents(document.id, all_chunks, embeddings)

                self.metadata_service.update_document_metadata(
                    document.id,
                    {
                        **parsed_doc.metadata,
                        "num_chunks": len(all_chunks),
                        "num_tables": len(parsed_doc.tables),
                        "num_text_chunks": len(text_chunks),
                        "num_table_chunks": len(parsed_doc.tables),
                    },
                )

                self.metadata_service.update_document_status(document.id, "completed")

                logger.info(
                    f"Document {document.id} processed successfully: "
                    f"{len(text_chunks)} text chunks, {len(parsed_doc.tables)} table chunks"
                )
                return document.id

            except Exception as e:
                self.metadata_service.update_document_status(document.id, "failed")
                logger.error(f"Failed to process document {document.id}: {str(e)}")
                raise DocumentProcessingError(
                    f"Document processing failed: {str(e)}"
                ) from e

        except Exception as e:
            logger.error(f"Error in process_document: {str(e)}")
            raise DocumentProcessingError(
                f"Failed to process document: {str(e)}"
            ) from e

    async def query_document(self, document_id: str, query: str) -> Dict[str, Any]:
        try:
            document = self.metadata_service.get_document(document_id)

            if document.status != "completed":
                raise DocumentProcessingError(
                    f"Document {document_id} is not ready for querying. Status: {document.status}"
                )

            query_embedding = self.embedder.embed_query(query)

            search_results = self.vector_db.search(
                document_id, query_embedding, top_k=self.settings.query_answering.top_k
            )

            if not search_results:
                return {
                    "answer": "No relevant information found in the document.",
                    "sources": [],
                    "confidence": 0.0,
                    "metadata": {},
                }

            if self.relevance_checker:
                is_relevant, confidence = (
                    self.relevance_checker.check_relevance_from_results(search_results)
                )
                if not is_relevant:
                    raise GatekeepingError(self.relevance_checker.rejection_message)

            from app.chunking.base import Chunk

            context_chunks = [
                Chunk(
                    text=result["text"],
                    chunk_id=result["chunk_id"],
                    metadata=result.get("metadata", {}),
                )
                for result in search_results
            ]

            query_response = await self.query_engine.answer(query, context_chunks)

            self.metadata_service.create_query(
                document_id=document_id,
                query_text=query,
                answer=query_response.answer,
                confidence=query_response.confidence,
            )

            return {
                "answer": query_response.answer,
                "sources": query_response.sources,
                "confidence": query_response.confidence,
                "metadata": query_response.metadata,
            }

        except GatekeepingError:
            raise
        except DocumentNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error querying document {document_id}: {str(e)}")
            raise DocumentProcessingError(f"Failed to query document: {str(e)}") from e

    def _extract_table_context(
        self, parsed_doc: ParsedDocument, table: Table, table_index: int
    ) -> Tuple[str, str, str, int]:
        page_number = table.metadata.get("page", 0) if table.metadata else 0

        section_title = ""
        context_before = ""
        context_after = ""

        if parsed_doc.structure.headers:
            for header in parsed_doc.structure.headers:
                header_page = header.get("page", 0)
                if header_page <= page_number:
                    header_text = header.get("text", "").strip()
                    if header_text and len(header_text) > 3:
                        section_title = header_text
                        break

        if parsed_doc.text:
            context_before, context_after = self._extract_text_around_table(
                parsed_doc.text, page_number, self.settings.chunking.chunk_size
            )

        return context_before, context_after, section_title, page_number

    def _extract_text_around_table(
        self, full_text: str, page_number: int, context_window: int
    ) -> Tuple[str, str]:
        pages = full_text.split("\f")

        if page_number > 0 and page_number <= len(pages):
            target_page_idx = page_number - 1
            target_page_text = pages[target_page_idx]

            page_length = len(target_page_text)
            midpoint = page_length // 2

            context_before = target_page_text[:midpoint][-context_window:]
            context_after = target_page_text[midpoint:][:context_window]

            if target_page_idx > 0:
                prev_page_text = pages[target_page_idx - 1][-context_window // 2 :]
                context_before = prev_page_text + context_before

            if target_page_idx < len(pages) - 1:
                next_page_text = pages[target_page_idx + 1][: context_window // 2]
                context_after = context_after + next_page_text
        else:
            paragraphs = full_text.split("\n\n")
            if paragraphs:
                context_before = (
                    "\n\n".join(paragraphs[-2:])[-context_window:]
                    if len(paragraphs) >= 2
                    else ""
                )
                context_after = (
                    "\n\n".join(paragraphs[:2])[:context_window]
                    if len(paragraphs) >= 2
                    else ""
                )
            else:
                context_before = ""
                context_after = ""

        return context_before.strip(), context_after.strip()

    async def delete_document(self, document_id: str) -> None:
        try:
            document = self.metadata_service.get_document(document_id)

            self.vector_db.delete_document(document_id)
            self.storage.delete_file(document.file_path)
            self.metadata_service.delete_document(document_id)

            logger.info(f"Document {document_id} deleted successfully")
        except DocumentNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            raise DocumentProcessingError(f"Failed to delete document: {str(e)}") from e
