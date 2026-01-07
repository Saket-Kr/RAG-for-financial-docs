from pathlib import Path
from typing import Any, Dict, List

from app.core.exceptions import ParserError
from app.parsers.base import BaseParser, DocumentStructure, ParsedDocument, Table
from app.parsers.table_normalizer import TableNormalizer


class UnstructuredParser(BaseParser):
    def __init__(self):
        try:
            from unstructured.partition.pdf import partition_pdf
            self.partition_pdf = partition_pdf
        except ImportError:
            raise ParserError("unstructured library not installed")

    def parse(self, file_path: str) -> ParsedDocument:
        try:
            elements = self.partition_pdf(
                filename=file_path,
                strategy="hi_res",
                infer_table_structure=True
            )
            
            text_parts = []
            tables = []
            headers = []
            sections = []
            page_num = 1
            
            for element in elements:
                if hasattr(element, "text"):
                    text_parts.append(element.text)
                
                if hasattr(element, "metadata") and hasattr(element.metadata, "page_number"):
                    page_num = element.metadata.page_number
                
                if element.category == "Title" or element.category == "NarrativeText":
                    if element.category == "Title":
                        headers.append({
                            "text": element.text,
                            "category": element.category,
                            "page": page_num
                        })
                    else:
                        sections.append({
                            "text": element.text,
                            "category": element.category
                        })
                
                if element.category == "Table":
                    table_data = self._extract_table_data(element)
                    if table_data:
                        table = TableNormalizer.normalize_from_unstructured(
                            raw_table_data=element,
                            page_num=page_num,
                            extra_metadata={"category": element.category}
                        )
                        tables.append(table)
            
            full_text = "\n\n".join(text_parts)
            
            structure = DocumentStructure(
                sections=sections,
                headers=headers,
                tables=tables,
                lists=[]
            )
            
            metadata = {
                "parser": "unstructured",
                "total_elements": len(elements),
                "total_tables": len(tables)
            }
            
            return ParsedDocument(
                text=full_text,
                tables=tables,
                metadata=metadata,
                structure=structure
            )
        except Exception as e:
            raise ParserError(f"Failed to parse PDF with unstructured: {str(e)}") from e

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".pdf"

    def _extract_table_data(self, element) -> List[List[str]]:
        try:
            if hasattr(element, "metadata") and hasattr(element.metadata, "text_as_html"):
                import html

                from bs4 import BeautifulSoup
                soup = BeautifulSoup(element.metadata.text_as_html, "html.parser")
                rows = []
                for tr in soup.find_all("tr"):
                    row = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                    if row:
                        rows.append(row)
                return rows
        except Exception:
            pass
        return []
