import pytest
from app.chunking.table_chunker import TableChunker
from app.parsers.base import Table


def test_table_chunker_basic():
    table = Table(
        data=[
            ["2024-01-15", "$5,000", "Paid"],
            ["2024-02-15", "$5,000", "Pending"]
        ],
        headers=["Date", "Amount", "Status"],
        metadata={"page": 3}
    )
    
    chunker = TableChunker(context_window=200)
    chunk = chunker.chunk_table(
        table=table,
        context_before="Payment schedule:",
        context_after="Late fees apply.",
        table_index=0,
        section_title="Payment Terms",
        page_number=3
    )
    
    assert chunk.metadata["chunk_type"] == "table"
    assert chunk.metadata["table_index"] == 0
    assert chunk.metadata["page_number"] == 3
    assert chunk.metadata["section_title"] == "Payment Terms"
    assert chunk.metadata["num_rows"] == 2
    assert chunk.metadata["has_headers"] is True
    assert "Date" in chunk.text
    assert "Amount" in chunk.text
    assert "Payment Terms" in chunk.text


def test_table_chunker_no_headers():
    table = Table(
        data=[
            ["2024-01-15", "$5,000", "Paid"],
            ["2024-02-15", "$5,000", "Pending"]
        ],
        headers=None,
        metadata={}
    )
    
    chunker = TableChunker()
    chunk = chunker.chunk_table(
        table=table,
        context_before="",
        context_after="",
        table_index=1,
        section_title="",
        page_number=1
    )
    
    assert chunk.metadata["has_headers"] is False
    assert "Row 1" in chunk.text
    assert "Row 2" in chunk.text


def test_table_chunker_metadata_enrichment():
    table = Table(
        data=[["Value1", "Value2"]],
        headers=["Column1", "Column2"],
        metadata={"page": 5}
    )
    
    chunker = TableChunker()
    chunk = chunker.chunk_table(
        table=table,
        context_before="Context before",
        context_after="Context after",
        table_index=2,
        section_title="Financial Summary",
        page_number=5
    )
    
    metadata = chunk.metadata
    assert metadata["chunk_type"] == "table"
    assert metadata["table_index"] == 2
    assert metadata["page_number"] == 5
    assert metadata["section_title"] == "Financial Summary"
    assert "table_headers" in metadata
    assert "table_summary" in metadata
    assert "keywords" in metadata
