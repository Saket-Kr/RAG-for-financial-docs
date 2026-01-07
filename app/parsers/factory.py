from typing import Dict, Type
from app.parsers.base import BaseParser
from app.parsers.pdfplumber_parser import PDFPlumberParser
from app.parsers.pymupdf_parser import PyMuPDFParser
from app.parsers.unstructured_parser import UnstructuredParser
from app.core.exceptions import ParserError


class ParserFactory:
    _parsers: Dict[str, Type[BaseParser]] = {
        "pdfplumber": PDFPlumberParser,
        "pymupdf": PyMuPDFParser,
        "unstructured": UnstructuredParser,
    }

    @classmethod
    def create_parser(cls, parser_type: str) -> BaseParser:
        if parser_type not in cls._parsers:
            raise ParserError(f"Unknown parser type: {parser_type}")
        
        parser_class = cls._parsers[parser_type]
        try:
            return parser_class()
        except Exception as e:
            raise ParserError(f"Failed to create parser {parser_type}: {str(e)}") from e

    @classmethod
    def register_parser(cls, name: str, parser_class: Type[BaseParser]) -> None:
        cls._parsers[name] = parser_class
