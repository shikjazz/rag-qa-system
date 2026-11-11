"""Document chunking strategies for RAG pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    """A text chunk with metadata."""

    text: str
    source: str
    chunk_id: int
    start_char: int = 0
    end_char: int = 0
    metadata: dict | None = None


class TextChunker:
    """Split documents into chunks for embedding and retrieval.

    Supports multiple chunking strategies:
    - Fixed-size with overlap
    - Sentence-based
    - Paragraph-based
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        strategy: str = "fixed",
    ):
        """Initialize the chunker.

        Args:
            chunk_size: Target characters per chunk.
            chunk_overlap: Character overlap between chunks.
            strategy: 'fixed', 'sentence', or 'paragraph'.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

    def chunk_text(self, text: str, source: str = "unknown") -> list[Chunk]:
        """Split text into chunks.

        Args:
            text: Text to split.
            source: Source identifier (filename, URL, etc).

        Returns:
            List of Chunk objects.
        """
        if self.strategy == "sentence":
            return self._chunk_by_sentences(text, source)
        elif self.strategy == "paragraph":
            return self._chunk_by_paragraphs(text, source)
        else:
            return self._chunk_fixed(text, source)

    def chunk_file(self, filepath: str | Path) -> list[Chunk]:
        """Read and chunk a file.

        Args:
            filepath: Path to the text file.

        Returns:
            List of Chunk objects.
        """
        filepath = Path(filepath)
        text = filepath.read_text(encoding="utf-8")
        return self.chunk_text(text, source=filepath.name)

    def chunk_directory(self, dirpath: str | Path, extensions: list[str] | None = None) -> list[Chunk]:
        """Chunk all files in a directory.

        Args:
            dirpath: Path to directory.
            extensions: File extensions to include (e.g. ['.txt', '.md']).

        Returns:
            Combined list of chunks from all files.
        """
        dirpath = Path(dirpath)
        extensions = extensions or [".txt", ".md"]
        chunks = []

        for ext in extensions:
            for filepath in sorted(dirpath.glob(f"*{ext}")):
                file_chunks = self.chunk_file(filepath)
                chunks.extend(file_chunks)

        return chunks

    def _chunk_fixed(self, text: str, source: str) -> list[Chunk]:
        """Fixed-size chunking with overlap."""
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at a sentence boundary
            if end < len(text):
                # Look for sentence boundary near the end
                boundary = self._find_boundary(text, end, window=100)
                if boundary > start:
                    end = boundary

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    source=source,
                    chunk_id=chunk_id,
                    start_char=start,
                    end_char=end,
                ))
                chunk_id += 1

            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks

    def _chunk_by_sentences(self, text: str, source: str) -> list[Chunk]:
        """Sentence-based chunking: group sentences up to chunk_size."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_text = ""
        start_char = 0
        chunk_id = 0

        for sentence in sentences:
            if len(current_text) + len(sentence) > self.chunk_size and current_text:
                chunks.append(Chunk(
                    text=current_text.strip(),
                    source=source,
                    chunk_id=chunk_id,
                    start_char=start_char,
                ))
                chunk_id += 1
                # Overlap: keep last sentence
                overlap_start = current_text.rfind(". ")
                if overlap_start > 0:
                    current_text = current_text[overlap_start + 2:]
                    start_char += overlap_start + 2
                else:
                    current_text = ""
                    start_char += len(current_text)

            current_text += sentence + " "

        if current_text.strip():
            chunks.append(Chunk(
                text=current_text.strip(),
                source=source,
                chunk_id=chunk_id,
                start_char=start_char,
            ))

        return chunks

    def _chunk_by_paragraphs(self, text: str, source: str) -> list[Chunk]:
        """Paragraph-based chunking."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_text = ""
        chunk_id = 0

        for para in paragraphs:
            if len(current_text) + len(para) > self.chunk_size and current_text:
                chunks.append(Chunk(
                    text=current_text.strip(),
                    source=source,
                    chunk_id=chunk_id,
                ))
                chunk_id += 1
                current_text = ""

            current_text += para + "\n\n"

        if current_text.strip():
            chunks.append(Chunk(
                text=current_text.strip(),
                source=source,
                chunk_id=chunk_id,
            ))

        return chunks

    @staticmethod
    def _find_boundary(text: str, pos: int, window: int = 100) -> int:
        """Find the nearest sentence boundary near pos."""
        search_start = max(0, pos - window)
        search_end = min(len(text), pos + window)
        region = text[search_start:search_end]

        # Find the last sentence-ending punctuation before pos
        for marker in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
            idx = region.rfind(marker, 0, pos - search_start + 1)
            if idx >= 0:
                return search_start + idx + len(marker)

        return pos
