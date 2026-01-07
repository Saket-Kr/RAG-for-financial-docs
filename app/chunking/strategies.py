import re
import uuid
from typing import Any, Dict, List, Optional

from app.chunking.base import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunk = Chunk(
                text=chunk_text,
                chunk_id=str(uuid.uuid4()),
                metadata={**(metadata or {}), "chunk_index": len(chunks)},
                start_index=start,
                end_index=end,
            )
            chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks


class SentenceChunker(BaseChunker):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=str(uuid.uuid4()),
                    metadata={**(metadata or {}), "chunk_index": len(chunks)},
                    start_index=None,
                    end_index=None,
                )
                chunks.append(chunk)

                overlap_sentences = self._get_overlap_sentences(
                    current_chunk, self.chunk_overlap
                )
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk = Chunk(
                text=chunk_text,
                chunk_id=str(uuid.uuid4()),
                metadata={**(metadata or {}), "chunk_index": len(chunks)},
                start_index=None,
                end_index=None,
            )
            chunks.append(chunk)

        return chunks

    def _get_overlap_sentences(
        self, sentences: List[str], overlap_size: int
    ) -> List[str]:
        overlap = []
        current_length = 0

        for sentence in reversed(sentences):
            if current_length + len(sentence) <= overlap_size:
                overlap.insert(0, sentence)
                current_length += len(sentence)
            else:
                break

        return overlap


class HierarchicalChunker(BaseChunker):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        sections = self._split_into_sections(text)
        chunks = []

        for section_idx, section in enumerate(sections):
            if len(section) <= self.chunk_size:
                chunk = Chunk(
                    text=section,
                    chunk_id=str(uuid.uuid4()),
                    metadata={
                        **(metadata or {}),
                        "chunk_index": len(chunks),
                        "section_index": section_idx,
                        "type": "section",
                    },
                )
                chunks.append(chunk)
            else:
                paragraphs = section.split("\n\n")
                current_chunk = []
                current_length = 0

                for para in paragraphs:
                    para_length = len(para)

                    if current_length + para_length > self.chunk_size and current_chunk:
                        chunk_text = "\n\n".join(current_chunk)
                        chunk = Chunk(
                            text=chunk_text,
                            chunk_id=str(uuid.uuid4()),
                            metadata={
                                **(metadata or {}),
                                "chunk_index": len(chunks),
                                "section_index": section_idx,
                                "type": "paragraph",
                            },
                        )
                        chunks.append(chunk)

                        overlap_paras = self._get_overlap_paragraphs(
                            current_chunk, self.chunk_overlap
                        )
                        current_chunk = overlap_paras + [para]
                        current_length = sum(len(p) for p in current_chunk)
                    else:
                        current_chunk.append(para)
                        current_length += para_length

                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunk = Chunk(
                        text=chunk_text,
                        chunk_id=str(uuid.uuid4()),
                        metadata={
                            **(metadata or {}),
                            "chunk_index": len(chunks),
                            "section_index": section_idx,
                            "type": "paragraph",
                        },
                    )
                    chunks.append(chunk)

        return chunks

    def _split_into_sections(self, text: str) -> List[str]:
        section_pattern = r"\n(?=[A-Z][^\n]{0,100}\n)"
        sections = re.split(section_pattern, text)
        return [s.strip() for s in sections if s.strip()]

    def _get_overlap_paragraphs(
        self, paragraphs: List[str], overlap_size: int
    ) -> List[str]:
        overlap = []
        current_length = 0

        for para in reversed(paragraphs):
            if current_length + len(para) <= overlap_size:
                overlap.insert(0, para)
                current_length += len(para)
            else:
                break

        return overlap


class SlidingWindowChunker(BaseChunker):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        chunks = []
        start = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            chunk = Chunk(
                text=chunk_text,
                chunk_id=str(uuid.uuid4()),
                metadata={**(metadata or {}), "chunk_index": len(chunks)},
                start_index=start,
                end_index=end,
            )
            chunks.append(chunk)

            start += step

        return chunks


class SemanticChunker(BaseChunker):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, embedder=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedder = embedder

    def chunk(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        if not self.embedder:
            sentence_chunker = SentenceChunker(self.chunk_size, self.chunk_overlap)
            return sentence_chunker.chunk(text, metadata)

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = []
        current_length = 0

        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)

            if current_length + sentence_length > self.chunk_size and current_chunk:
                if i < len(sentences) - 1:
                    similarity = self._calculate_similarity(
                        " ".join(current_chunk), sentence
                    )

                    if similarity < 0.7:
                        chunk_text = " ".join(current_chunk)
                        chunk = Chunk(
                            text=chunk_text,
                            chunk_id=str(uuid.uuid4()),
                            metadata={**(metadata or {}), "chunk_index": len(chunks)},
                        )
                        chunks.append(chunk)

                        overlap_sentences = self._get_overlap_sentences(
                            current_chunk, self.chunk_overlap
                        )
                        current_chunk = overlap_sentences + [sentence]
                        current_length = sum(len(s) for s in current_chunk)
                    else:
                        current_chunk.append(sentence)
                        current_length += sentence_length
                else:
                    chunk_text = " ".join(current_chunk)
                    chunk = Chunk(
                        text=chunk_text,
                        chunk_id=str(uuid.uuid4()),
                        metadata={**(metadata or {}), "chunk_index": len(chunks)},
                    )
                    chunks.append(chunk)
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk = Chunk(
                text=chunk_text,
                chunk_id=str(uuid.uuid4()),
                metadata={**(metadata or {}), "chunk_index": len(chunks)},
            )
            chunks.append(chunk)

        return chunks

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        try:
            emb1 = self.embedder.embed_query(text1)
            emb2 = self.embedder.embed_query(text2)
            import numpy as np

            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )
            return float(similarity)
        except Exception:
            return 0.5

    def _get_overlap_sentences(
        self, sentences: List[str], overlap_size: int
    ) -> List[str]:
        overlap = []
        current_length = 0

        for sentence in reversed(sentences):
            if current_length + len(sentence) <= overlap_size:
                overlap.insert(0, sentence)
                current_length += len(sentence)
            else:
                break

        return overlap
