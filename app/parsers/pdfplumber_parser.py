from pathlib import Path
from typing import Any, Dict, List

import pdfplumber

from app.core.exceptions import ParserError
from app.parsers.base import BaseParser, DocumentStructure, ParsedDocument, Table
from app.parsers.table_normalizer import TableNormalizer


class PDFPlumberParser(BaseParser):
    def parse(self, file_path: str) -> ParsedDocument:
        try:
            text_parts = []
            tables = []
            headers = []
            sections = []

            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    text_parts.append(page_text)

                    page_tables = page.extract_tables()
                    for table_data in page_tables:
                        if table_data:
                            table = TableNormalizer.normalize_from_pdfplumber(
                                raw_table_data=table_data,
                                page_num=page_num,
                                extra_metadata={},
                            )
                            tables.append(table)

                    page_headers = self._extract_headers(page)
                    headers.extend([{**h, "page": page_num} for h in page_headers])

            full_text = "\n\n".join(text_parts)

            structure = DocumentStructure(
                sections=sections, headers=headers, tables=tables, lists=[]
            )

            metadata = {
                "parser": "pdfplumber",
                "total_pages": len(text_parts),
                "total_tables": len(tables),
            }

            return ParsedDocument(
                text=full_text, tables=tables, metadata=metadata, structure=structure
            )
        except Exception as e:
            raise ParserError(f"Failed to parse PDF with pdfplumber: {str(e)}") from e

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".pdf"

    def _extract_headers(self, page) -> List[Dict[str, Any]]:
        headers = []
        chars = page.chars
        if not chars:
            return headers

        for char in chars:
            if char.get("size", 0) > 12:
                headers.append(
                    {
                        "text": char.get("text", ""),
                        "size": char.get("size", 0),
                        "top": char.get("top", 0),
                    }
                )
        return headers
