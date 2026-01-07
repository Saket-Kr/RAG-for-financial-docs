import pytest

from app.core.exceptions import ParserError
from app.parsers.factory import ParserFactory


def test_parser_factory():
    parser = ParserFactory.create_parser("pdfplumber")
    assert parser is not None


def test_parser_factory_invalid():
    with pytest.raises(ParserError):
        ParserFactory.create_parser("invalid_parser")
