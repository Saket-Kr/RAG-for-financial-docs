from unittest.mock import MagicMock, Mock, patch

import pytest
from app.core.config import Settings
from app.core.exceptions import DocumentProcessingError
from app.parsers.base import DocumentStructure, ParsedDocument, Table
from app.services.document_service import DocumentService
from app.services.metadata_service import MetadataService
from app.services.storage_service import LocalStorage


@pytest.fixture
def mock_settings():
    settings = Mock(spec=Settings)
    settings.pdf_parser = Mock()
    settings.pdf_parser.type = "pdfplumber"
    settings.embeddings = Mock()
    settings.embeddings.type = "nomic"
    settings.embeddings.model_name = "nomic-embed-text-v1"
    settings.embeddings.device = "cpu"
    settings.embeddings.batch_size = 32
    settings.vector_db = Mock()
    settings.vector_db.type = "chroma"
    settings.vector_db.persist_directory = "./test_vector_db"
    settings.vector_db.collection_name = "test_collection"
    settings.vector_db.distance_metric = "cosine"
    settings.query_answering = Mock()
    settings.query_answering.strategy = "rag"
    settings.query_answering.top_k = 5
    settings.query_answering.llm = Mock()
    settings.query_answering.llm.base_url = "http://localhost:11434"
    settings.query_answering.llm.model = "mistral"
    settings.query_answering.llm.temperature = 0.0
    settings.query_answering.llm.max_tokens = 512
    settings.query_answering.llm.timeout = 30
    settings.gatekeeping = Mock()
    settings.gatekeeping.enabled = False
    settings.chunking = Mock()
    settings.chunking.strategy = "fixed"
    settings.chunking.chunk_size = 500
    settings.chunking.chunk_overlap = 50
    return settings


@pytest.fixture
def mock_parsed_document_with_tables():
    table1 = Table(
        data=[
            ["2024-01-15", "$5,000", "Paid"],
            ["2024-02-15", "$5,000", "Pending"]
        ],
        headers=["Date", "Amount", "Status"],
        metadata={"page": 1}
    )
    
    table2 = Table(
        data=[
            ["Principal", "$100,000"],
            ["Interest Rate", "3.5%"]
        ],
        headers=["Item", "Value"],
        metadata={"page": 2}
    )
    
    structure = DocumentStructure(
        sections=[],
        headers=[{"text": "Payment Schedule", "page": 1}],
        tables=[table1, table2],
        lists=[]
    )
    
    return ParsedDocument(
        text="This is sample document text.\n\nPayment schedule details below.",
        tables=[table1, table2],
        metadata={"parser": "pdfplumber", "total_pages": 2, "total_tables": 2},
        structure=structure
    )


@patch("app.services.document_service.ParserFactory")
@patch("app.services.document_service.ChunkingFactory")
@patch("app.services.document_service.EmbeddingFactory")
@patch("app.services.document_service.VectorDBFactory")
@patch("app.services.document_service.QueryEngineFactory")
@patch("app.services.document_service.OllamaClient")
def test_process_document_with_tables(
    mock_ollama,
    mock_query_factory,
    mock_vector_factory,
    mock_embedding_factory,
    mock_chunking_factory,
    mock_parser_factory,
    mock_settings,
    mock_parsed_document_with_tables
):
    mock_metadata_service = Mock(spec=MetadataService)
    mock_storage = Mock(spec=LocalStorage)
    
    mock_document = Mock()
    mock_document.id = "test-doc-123"
    mock_document.file_path = "/tmp/test.pdf"
    mock_metadata_service.create_document.return_value = mock_document
    
    mock_parser = Mock()
    mock_parser.parse.return_value = mock_parsed_document_with_tables
    mock_parser_factory.create_parser.return_value = mock_parser
    
    mock_chunker = Mock()
    mock_chunker.chunk.return_value = [
        Mock(text="Sample text chunk", chunk_id="chunk-1", metadata={"chunk_type": "text"})
    ]
    mock_chunking_factory.create_chunker.return_value = mock_chunker
    
    mock_embedder = Mock()
    mock_embedder.embed_texts.return_value = [[0.1] * 768] * 3
    mock_embedding_factory.create_embedder.return_value = mock_embedder
    
    mock_vector_db = Mock()
    mock_vector_factory.create_db.return_value = mock_vector_db
    
    mock_ollama_client = Mock()
    mock_ollama.return_value = mock_ollama_client
    mock_query_engine = Mock()
    mock_query_factory.create_engine.return_value = mock_query_engine
    
    service = DocumentService(mock_settings, mock_metadata_service, mock_storage)
    
    import asyncio
    result = asyncio.run(service.process_document(b"fake pdf content", "test.pdf"))
    
    assert result == "test-doc-123"
    
    mock_chunker.chunk.assert_called_once()
    assert mock_embedder.embed_texts.call_count == 1
    
    call_args = mock_embedder.embed_texts.call_args[0][0]
    assert len(call_args) == 3
    
    mock_vector_db.add_documents.assert_called_once()
    add_docs_call = mock_vector_db.add_documents.call_args
    chunks = add_docs_call[0][1]
    
    table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
    assert len(table_chunks) == 2
    
    mock_metadata_service.update_document_metadata.assert_called()
    metadata_call = mock_metadata_service.update_document_metadata.call_args[0][1]
    assert metadata_call["num_tables"] == 2
    assert metadata_call["num_table_chunks"] == 2


def test_extract_table_context(mock_settings):
    mock_metadata_service = Mock(spec=MetadataService)
    mock_storage = Mock(spec=LocalStorage)
    
    service = DocumentService(mock_settings, mock_metadata_service, mock_storage)
    
    table = Table(
        data=[["Row1", "Row2"]],
        headers=["Col1", "Col2"],
        metadata={"page": 2}
    )
    
    parsed_doc = ParsedDocument(
        text="Page 1 text\fPage 2 text with table\fPage 3 text",
        tables=[table],
        metadata={},
        structure=DocumentStructure(
            sections=[],
            headers=[{"text": "Section Title", "page": 1}],
            tables=[table],
            lists=[]
        )
    )
    
    context_before, context_after, section_title, page_number = service._extract_table_context(
        parsed_doc, table, 0
    )
    
    assert page_number == 2
    assert section_title == "Section Title"
    assert isinstance(context_before, str)
    assert isinstance(context_after, str)
