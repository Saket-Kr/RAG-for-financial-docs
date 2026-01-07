import pytest
from app.parsers.factory import ParserFactory
from app.core.exceptions import ParserError


def test_parser_factory():
    parser = ParserFactory.create_parser("pdfplumber")
    assert parser is not None


def test_parser_factory_invalid():
    with pytest.raises(ParserError):
        ParserFactory.create_parser("invalid_parser")
