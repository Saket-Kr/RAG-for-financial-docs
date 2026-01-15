import re
import uuid
from typing import Any, Dict, List

from app.chunking.base import Chunk
from app.parsers.base import Table


class TableChunker:
    def __init__(self, context_window: int = 200):
        self.context_window = context_window

    def chunk_table(
        self,
        table: Table,
        context_before: str = "",
        context_after: str = "",
        table_index: int = 0,
        section_title: str = "",
        page_number: int = 0,
    ) -> Chunk:
        table_text = self._table_to_hybrid_format(table, section_title)

        full_text = self._build_contextual_text(
            context_before, table_text, context_after, section_title
        )

        metadata = self._enrich_metadata(
            table,
            table_index,
            page_number,
            section_title,
            context_before,
            context_after,
        )

        return Chunk(text=full_text, chunk_id=str(uuid.uuid4()), metadata=metadata)

    def _table_to_hybrid_format(self, table: Table, section_title: str = "") -> str:
        """
        Format table data into a structured text representation.
        
        Emphasizes headers, column names, and section titles for better semantic search.
        """
        if not table.data:
            return "Empty table with no data."

        num_rows = len(table.data)
        lines = []

        # Emphasize section title if available
        if section_title:
            lines.append(f"Table: {section_title}")
            lines.append("")

        # Header section - make column names very prominent
        if table.headers:
            header_text = ", ".join(table.headers)
            lines.append(f"Table with columns: {header_text}")
            lines.append(f"Column names: {header_text}")
            lines.append(f"This table contains {num_rows} row(s) of data:")
        else:
            lines.append(f"Table data ({num_rows} row(s)):")

        lines.append("")

        # Format each row with column names emphasized
        for row_idx, row in enumerate(table.data, 1):
            if not row:
                continue

            if table.headers:
                row_parts = []
                for col_idx, value in enumerate(row):
                    if col_idx < len(table.headers):
                        header_name = table.headers[col_idx]
                        cell_value = str(value).strip() if value else ""
                        row_parts.append(f"{header_name}: {cell_value}")
                    else:
                        row_parts.append(str(value).strip() if value else "")
                row_text = ", ".join(row_parts)
            else:
                row_text = " | ".join(
                    str(cell).strip() if cell else "" for cell in row
                )

            lines.append(f"Row {row_idx}: {row_text}")

        # Summary footer with row count and column names
        lines.append("")
        if table.headers:
            header_summary = ", ".join(table.headers)
            if section_title:
                lines.append(
                    f"Total: {num_rows} row(s) in this {section_title} table with columns: {header_summary}."
                )
            else:
                lines.append(
                    f"Total: {num_rows} row(s) in this table with columns: {header_summary}."
                )
        else:
            lines.append(f"Total: {num_rows} row(s) in this table.")

        return "\n".join(lines)

    def _build_contextual_text(
        self,
        context_before: str,
        table_text: str,
        context_after: str,
        section_title: str,
    ) -> str:
        """
        Build the full contextual text by combining section, context, and table.
        
        Emphasizes section title and makes it prominent at the start.
        """
        parts = []

        # Make section title VERY prominent - repeat it at the start
        if section_title:
            parts.append(f"Section: {section_title}")
            parts.append(f"Table: {section_title}")

        if context_before:
            truncated_before = (
                context_before[-self.context_window :]
                if len(context_before) > self.context_window
                else context_before
            )
            parts.append(f"Context: {truncated_before}")

        parts.append(table_text)

        if context_after:
            truncated_after = context_after[: self.context_window]
            parts.append(f"Following context: {truncated_after}")

        return "\n\n".join(parts)

    def _enrich_metadata(
        self,
        table: Table,
        table_index: int,
        page_number: int,
        section_title: str,
        context_before: str,
        context_after: str,
    ) -> Dict[str, Any]:
        """
        Create rich metadata for the table chunk.

        Args:
            table: Table object
            table_index: Index of the table in the document
            page_number: Page number where table appears
            section_title: Title of the section
            context_before: Text before the table
            context_after: Text after the table

        Returns:
            Dictionary of metadata attributes
        """
        metadata: Dict[str, Any] = {
            "chunk_type": "table",
            "table_index": table_index,
            "page_number": page_number,
            "num_rows": len(table.data),
            "num_columns": self._calculate_column_count(table),
            "has_headers": table.headers is not None,
        }

        if table.headers:
            metadata["table_headers"] = table.headers
            metadata["table_columns"] = table.headers

        if section_title:
            metadata["section_title"] = section_title
            metadata["keywords"] = self._extract_keywords(section_title)

        if context_before:
            metadata["context_before"] = (
                context_before[-100:] if len(context_before) > 100 else context_before
            )

        if context_after:
            metadata["context_after"] = (
                context_after[:100] if len(context_after) > 100 else context_after
            )

        metadata["table_summary"] = self._generate_table_summary(table)

        if table.metadata:
            metadata.update(table.metadata)

        return metadata

    def _calculate_column_count(self, table: Table) -> int:
        """
        Calculate the number of columns in the table.

        Args:
            table: Table object

        Returns:
            Number of columns (from headers or first row)
        """
        if table.headers:
            return len(table.headers)
        if table.data and len(table.data) > 0:
            return len(table.data[0])
        return 0

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract meaningful keywords from a text string.

        Args:
            text: Text to extract keywords from

        Returns:
            List of keywords (words longer than 3 characters, max 10)
        """
        words = re.findall(r"\b\w+\b", text.lower())
        return [w for w in words if len(w) > 3][:10]

    def _generate_table_summary(self, table: Table) -> str:
        """
        Generate a human-readable summary of the table.

        Args:
            table: Table object

        Returns:
            Summary string describing the table
        """
        if not table.data:
            return "Empty table"

        num_rows = len(table.data)

        if table.headers:
            header_preview = ", ".join(table.headers[:3])
            if len(table.headers) > 3:
                header_preview += f" (+{len(table.headers) - 3} more columns)"
            return f"Table with {num_rows} row(s) containing {header_preview} data"
        else:
            num_cols = len(table.data[0]) if table.data else 0
            return f"Table with {num_rows} row(s) and {num_cols} column(s)"
