from typing import Any, Dict, List, Optional

from app.core.exceptions import ParserError
from app.parsers.base import Table


class TableNormalizer:
    @staticmethod
    def normalize_from_pdfplumber(
        raw_table_data: List[List[str]],
        page_num: int,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Table:
        if not raw_table_data:
            return Table(
                data=[],
                headers=None,
                metadata={
                    "page": page_num,
                    "source_parser": "pdfplumber",
                    **(extra_metadata or {})
                }
            )
        
        headers = None
        data = raw_table_data
        
        if len(raw_table_data) > 0:
            first_row = raw_table_data[0]
            if first_row and all(isinstance(cell, str) for cell in first_row):
                potential_headers = first_row
                if TableNormalizer._looks_like_header_row(potential_headers, raw_table_data[1:] if len(raw_table_data) > 1 else []):
                    headers = [str(cell).strip() if cell else "" for cell in potential_headers]
                    data = raw_table_data[1:]
        
        normalized_data = [
            [str(cell).strip() if cell else "" for cell in row]
            for row in data
        ]
        
        metadata = {
            "page": page_num,
            "source_parser": "pdfplumber",
            "raw_row_count": len(raw_table_data),
            **(extra_metadata or {})
        }
        
        return Table(
            data=normalized_data,
            headers=headers,
            metadata=metadata
        )

    @staticmethod
    def normalize_from_pymupdf(
        raw_table_data: Any,
        page_num: int,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Table:
        if isinstance(raw_table_data, dict):
            data = raw_table_data.get("data", [])
            headers = raw_table_data.get("headers")
        elif isinstance(raw_table_data, list):
            if not raw_table_data:
                return Table(
                    data=[],
                    headers=None,
                    metadata={
                        "page": page_num,
                        "source_parser": "pymupdf",
                        **(extra_metadata or {})
                    }
                )
            
            first_row = raw_table_data[0] if raw_table_data else []
            if TableNormalizer._looks_like_header_row(first_row, raw_table_data[1:] if len(raw_table_data) > 1 else []):
                headers = [str(cell).strip() if cell else "" for cell in first_row]
                data = raw_table_data[1:]
            else:
                headers = None
                data = raw_table_data
        else:
            data = []
            headers = None
        
        normalized_data = [
            [str(cell).strip() if cell else "" for cell in row]
            for row in data
        ]
        
        metadata = {
            "page": page_num,
            "source_parser": "pymupdf",
            "raw_row_count": len(data) + (1 if headers else 0),
            **(extra_metadata or {})
        }
        
        return Table(
            data=normalized_data,
            headers=headers,
            metadata=metadata
        )

    @staticmethod
    def normalize_from_unstructured(
        raw_table_data: Any,
        page_num: int,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Table:
        try:
            from bs4 import BeautifulSoup
            
            normalized_data = []
            headers = None
            
            if hasattr(raw_table_data, "metadata") and hasattr(raw_table_data.metadata, "text_as_html"):
                html_content = raw_table_data.metadata.text_as_html
                soup = BeautifulSoup(html_content, "html.parser")
                
                for tr in soup.find_all("tr"):
                    row = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                    if row:
                        if headers is None and tr.find("th"):
                            headers = row
                        else:
                            normalized_data.append(row)
            elif isinstance(raw_table_data, list):
                if not raw_table_data:
                    return Table(
                        data=[],
                        headers=None,
                        metadata={
                            "page": page_num,
                            "source_parser": "unstructured",
                            **(extra_metadata or {})
                        }
                    )
                
                first_row = raw_table_data[0] if raw_table_data else []
                if TableNormalizer._looks_like_header_row(first_row, raw_table_data[1:] if len(raw_table_data) > 1 else []):
                    headers = [str(cell).strip() if cell else "" for cell in first_row]
                    normalized_data = raw_table_data[1:]
                else:
                    headers = None
                    normalized_data = raw_table_data
            else:
                normalized_data = []
                headers = None
            
            normalized_data = [
                [str(cell).strip() if cell else "" for cell in row]
                for row in normalized_data
            ]
            
            metadata = {
                "page": page_num,
                "source_parser": "unstructured",
                "raw_row_count": len(normalized_data) + (1 if headers else 0),
                **(extra_metadata or {})
            }
            
            return Table(
                data=normalized_data,
                headers=headers,
                metadata=metadata
            )
        except Exception as e:
            raise ParserError(f"Failed to normalize unstructured table: {str(e)}") from e

    @staticmethod
    def _looks_like_header_row(
        potential_header_row: List[str],
        data_rows: List[List[str]]
    ) -> bool:
        if not potential_header_row or not data_rows:
            return False
        
        header_cells = [str(cell).strip().lower() for cell in potential_header_row if cell]
        if not header_cells:
            return False
        
        if len(header_cells) < 2:
            return False
        
        header_text = " ".join(header_cells)
        
        common_header_words = {
            "date", "amount", "total", "name", "description", "status",
            "payment", "due", "balance", "principal", "interest", "rate",
            "account", "number", "id", "type", "category", "item"
        }
        
        header_word_count = sum(1 for word in common_header_words if word in header_text)
        
        if header_word_count >= 1:
            return True
        
        if all(len(str(cell).strip()) < 50 for cell in potential_header_row):
            if all(not str(cell).strip().replace(".", "").replace(",", "").isdigit() for cell in potential_header_row):
                return True
        
        return False
