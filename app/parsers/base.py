from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Table:
    data: List[List[str]]
    headers: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DocumentStructure:
    sections: List[Dict[str, Any]]
    headers: List[Dict[str, Any]]
    tables: List[Table]
    lists: List[Dict[str, Any]]


@dataclass
class ParsedDocument:
    text: str
    tables: List[Table]
    metadata: Dict[str, Any]
    structure: DocumentStructure


class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        pass

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        pass
