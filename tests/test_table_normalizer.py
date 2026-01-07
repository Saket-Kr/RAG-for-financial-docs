import pytest
from app.parsers.base import Table
from app.parsers.table_normalizer import TableNormalizer


def test_normalize_from_pdfplumber():
    raw_data = [
        ["Date", "Amount", "Status"],
        ["2024-01-15", "$5,000", "Paid"],
        ["2024-02-15", "$5,000", "Pending"]
    ]
    
    table = TableNormalizer.normalize_from_pdfplumber(
        raw_table_data=raw_data,
        page_num=3,
        extra_metadata={}
    )
    
    assert table.headers == ["Date", "Amount", "Status"]
    assert len(table.data) == 2
    assert table.data[0] == ["2024-01-15", "$5,000", "Paid"]
    assert table.metadata["page"] == 3
    assert table.metadata["source_parser"] == "pdfplumber"


def test_normalize_from_pdfplumber_no_headers():
    raw_data = [
        ["2024-01-15", "$5,000", "Paid"],
        ["2024-02-15", "$5,000", "Pending"]
    ]
    
    table = TableNormalizer.normalize_from_pdfplumber(
        raw_table_data=raw_data,
        page_num=1
    )
    
    assert table.headers is None
    assert len(table.data) == 2


def test_normalize_from_pymupdf():
    raw_data = [
        ["Date", "Amount", "Status"],
        ["2024-01-15", "$5,000", "Paid"]
    ]
    
    table = TableNormalizer.normalize_from_pymupdf(
        raw_table_data=raw_data,
        page_num=2
    )
    
    assert table.headers == ["Date", "Amount", "Status"]
    assert len(table.data) == 1
    assert table.metadata["source_parser"] == "pymupdf"


def test_normalize_from_pymupdf_dict_format():
    raw_data = {
        "data": [
            ["2024-01-15", "$5,000", "Paid"]
        ],
        "headers": ["Date", "Amount", "Status"]
    }
    
    table = TableNormalizer.normalize_from_pymupdf(
        raw_table_data=raw_data,
        page_num=1
    )
    
    assert table.headers == ["Date", "Amount", "Status"]
    assert len(table.data) == 1
