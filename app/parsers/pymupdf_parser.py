from pathlib import Path
from typing import Any, Dict, List

import fitz
from app.core.exceptions import ParserError
from app.parsers.base import BaseParser, DocumentStructure, ParsedDocument, Table
from app.parsers.table_normalizer import TableNormalizer


class PyMuPDFParser(BaseParser):
    def parse(self, file_path: str) -> ParsedDocument:
        try:
            doc = fitz.open(file_path)
            text_parts = []
            tables = []
            headers = []
            sections = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                text_parts.append(page_text)
                
                page_tables = self._extract_tables(page)
                for raw_table in page_tables:
                    table = TableNormalizer.normalize_from_pymupdf(
                        raw_table_data=raw_table,
                        page_num=page_num + 1,
                        extra_metadata={}
                    )
                    tables.append(table)
                
                page_headers = self._extract_headers(page)
                headers.extend([{**h, "page": page_num + 1} for h in page_headers])
            
            doc.close()
            
            full_text = "\n\n".join(text_parts)
            
            structure = DocumentStructure(
                sections=sections,
                headers=headers,
                tables=tables,
                lists=[]
            )
            
            metadata = {
                "parser": "pymupdf",
                "total_pages": len(text_parts),
                "total_tables": len(tables)
            }
            
            return ParsedDocument(
                text=full_text,
                tables=tables,
                metadata=metadata,
                structure=structure
            )
        except Exception as e:
            raise ParserError(f"Failed to parse PDF with PyMuPDF: {str(e)}") from e

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".pdf"

    def _extract_tables(self, page) -> List[Dict[str, Any]]:
        tables = []
        try:
            tabs = page.find_tables()
            for tab in tabs:
                table_data = tab.extract()
                if table_data:
                    tables.append({"data": table_data})
        except Exception:
            pass
        return tables

    def _extract_headers(self, page) -> List[Dict[str, Any]]:
        headers = []
        try:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            if span.get("size", 0) > 12:
                                headers.append({
                                    "text": span.get("text", ""),
                                    "size": span.get("size", 0)
                                })
        except Exception:
            pass
        return headers
