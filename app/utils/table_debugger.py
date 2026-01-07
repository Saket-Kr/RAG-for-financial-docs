import logging
from typing import Any, Dict, List

from app.core.config import Settings
from app.embeddings.factory import EmbeddingFactory
from app.parsers.factory import ParserFactory
from app.services.document_service import DocumentService
from app.services.metadata_service import MetadataService
from app.vector_db.factory import VectorDBFactory

logger = logging.getLogger(__name__)


class TableDebugger:
    def __init__(
        self,
        metadata_service: MetadataService,
        document_service: DocumentService,
        settings: Settings,
    ):
        self.metadata_service = metadata_service
        self.document_service = document_service
        self.settings = settings

    def log_extracted_tables(self, document_id: str) -> None:
        document = self.metadata_service.get_document(document_id)
        parser = ParserFactory.create_parser(self.settings.pdf_parser.type)
        parsed_doc = parser.parse(document.file_path)

        logger.info(f"=== Table Extraction Report for Document {document_id} ===")
        logger.info(f"Total tables found: {len(parsed_doc.tables)}")

        for idx, table in enumerate(parsed_doc.tables):
            logger.info(f"\nTable {idx + 1}:")
            logger.info(
                f"  Page: {table.metadata.get('page', 'Unknown') if table.metadata else 'Unknown'}"
            )
            logger.info(f"  Rows: {len(table.data)}")
            logger.info(f"  Headers: {table.headers}")
            if table.data:
                logger.info(f"  Sample row: {table.data[0]}")

        logger.info("=" * 60)

    def log_table_chunks(self, document_id: str) -> None:
        vector_db = VectorDBFactory.create_db(
            self.settings.vector_db.type,
            self.settings.vector_db.persist_directory,
            self.settings.vector_db.collection_name,
            self.settings.vector_db.distance_metric,
        )

        embedder = EmbeddingFactory.create_embedder(
            self.settings.embeddings.type,
            self.settings.embeddings.model_name,
            self.settings.embeddings.device,
            self.settings.embeddings.batch_size,
        )

        dummy_query = "table"
        query_embedding = embedder.embed_query(dummy_query)

        search_results = vector_db.search(document_id, query_embedding, top_k=100)

        table_chunks = [
            r
            for r in search_results
            if r.get("metadata", {}).get("chunk_type") == "table"
        ]

        logger.info(f"=== Table Chunks in Vector DB for Document {document_id} ===")
        logger.info(f"Total table chunks found: {len(table_chunks)}")

        for idx, chunk in enumerate(table_chunks):
            metadata = chunk.get("metadata", {})
            logger.info(f"\nTable Chunk {idx + 1}:")
            logger.info(f"  Chunk ID: {chunk.get('chunk_id')}")
            logger.info(f"  Table Index: {metadata.get('table_index')}")
            logger.info(f"  Page: {metadata.get('page_number')}")
            logger.info(f"  Section: {metadata.get('section_title', 'N/A')}")
            logger.info(f"  Headers: {metadata.get('table_headers', 'N/A')}")
            logger.info(f"  Rows: {metadata.get('num_rows')}")
            logger.info(f"  Text preview: {chunk.get('text', '')[:200]}...")

        logger.info("=" * 60)

    def simulate_table_query(
        self, document_id: str, query: str, top_k: int = 5
    ) -> Dict[str, Any]:
        embedder = EmbeddingFactory.create_embedder(
            self.settings.embeddings.type,
            self.settings.embeddings.model_name,
            self.settings.embeddings.device,
            self.settings.embeddings.batch_size,
        )

        vector_db = VectorDBFactory.create_db(
            self.settings.vector_db.type,
            self.settings.vector_db.persist_directory,
            self.settings.vector_db.collection_name,
            self.settings.vector_db.distance_metric,
        )

        query_embedding = embedder.embed_query(query)
        search_results = vector_db.search(document_id, query_embedding, top_k=top_k)

        table_chunks = [
            r
            for r in search_results
            if r.get("metadata", {}).get("chunk_type") == "table"
        ]
        text_chunks = [
            r
            for r in search_results
            if r.get("metadata", {}).get("chunk_type") != "table"
        ]

        logger.info(f"=== Query Simulation: '{query}' ===")
        logger.info(f"Total results: {len(search_results)}")
        logger.info(f"Table chunks: {len(table_chunks)}")
        logger.info(f"Text chunks: {len(text_chunks)}")

        for idx, result in enumerate(search_results, 1):
            chunk_type = result.get("metadata", {}).get("chunk_type", "unknown")
            logger.info(f"\nResult {idx} ({chunk_type}):")
            logger.info(f"  Distance: {result.get('distance', 'N/A')}")
            logger.info(f"  Preview: {result.get('text', '')[:150]}...")
            if chunk_type == "table":
                metadata = result.get("metadata", {})
                logger.info(f"  Table Index: {metadata.get('table_index')}")
                logger.info(f"  Headers: {metadata.get('table_headers', 'N/A')}")

        logger.info("=" * 60)

        return {
            "total_results": len(search_results),
            "table_chunks": len(table_chunks),
            "text_chunks": len(text_chunks),
            "results": search_results,
        }
