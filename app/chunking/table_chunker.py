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
        table_text = self._table_to_hybrid_format(table)

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

    def _table_to_hybrid_format(self, table: Table) -> str:
        lines = []

        if not table.data:
            return "Empty table with no data."

        if table.headers:
            header_text = ", ".join(table.headers)
            lines.append(f"Table with columns: {header_text}")
            lines.append("The table contains the following data:")
        else:
            lines.append("Table data:")

        for row_idx, row in enumerate(table.data, 1):
            if not row:
                continue

            if table.headers:
                row_parts = []
                for col_idx, value in enumerate(row):
                    if col_idx < len(table.headers):
                        row_parts.append(f"{table.headers[col_idx]}: {value}")
                    else:
                        row_parts.append(str(value))
                row_text = ", ".join(row_parts)
            else:
                row_text = " | ".join(str(cell) for cell in row)

            lines.append(f"Row {row_idx}: {row_text}")

        return "\n".join(lines) if lines else "Empty table."

    def _build_contextual_text(
        self,
        context_before: str,
        table_text: str,
        context_after: str,
        section_title: str,
    ) -> str:
        parts = []

        if section_title:
            parts.append(f"Section: {section_title}")

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
        metadata = {
            "chunk_type": "table",
            "table_index": table_index,
            "page_number": page_number,
            "num_rows": len(table.data),
            "num_columns": (
                len(table.headers)
                if table.headers
                else (len(table.data[0]) if table.data else 0)
            ),
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

    def _extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        return [w for w in words if len(w) > 3][:10]

    def _generate_table_summary(self, table: Table) -> str:
        if not table.data:
            return "Empty table"

        if table.headers:
            header_preview = ", ".join(table.headers[:3])
            if len(table.headers) > 3:
                header_preview += "..."
            return f"Table with {len(table.data)} rows containing {header_preview} data"
        else:
            return f"Table with {len(table.data)} rows"
