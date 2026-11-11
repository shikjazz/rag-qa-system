"""FAISS-based vector retrieval for RAG pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore[assignment]

from rag_qa.chunker import Chunk


@dataclass
class RetrievalResult:
    """A retrieved chunk with similarity score."""

    chunk: Chunk
    score: float
    rank: int


class Retriever:
    """FAISS-based vector similarity search.

    Stores chunk embeddings in a FAISS index for fast
    nearest-neighbor retrieval at query time.
    """

    def __init__(self, embedding_dim: int):
        """Initialize the retriever.

        Args:
            embedding_dim: Dimension of embedding vectors.
        """
        if faiss is None:
            raise ImportError(
                "faiss is required. Install with: pip install faiss-cpu"
            )

        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine sim for normalized vecs)
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks and their embeddings to the index.

        Args:
            chunks: List of text chunks.
            embeddings: Corresponding embedding matrix.
        """
        assert len(chunks) == embeddings.shape[0], (
            f"Mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings"
        )

        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[RetrievalResult]:
        """Search for the most similar chunks.

        Args:
            query_embedding: Query vector (1 x dim).
            top_k: Number of results to return.
            threshold: Minimum similarity score.

        Returns:
            List of RetrievalResult objects, sorted by score.
        """
        if self.index.ntotal == 0:
            return []

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(
            query_embedding.astype(np.float32).reshape(1, -1),
            k,
        )

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or score < threshold:
                continue
            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                score=float(score),
                rank=rank,
            ))

        return results

    def save(self, path: str | Path) -> None:
        """Persist the index and metadata to disk.

        Args:
            path: Directory to save index files.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))

        metadata = []
        for chunk in self.chunks:
            metadata.append({
                "text": chunk.text,
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            })

        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def load(self, path: str | Path) -> None:
        """Load a persisted index from disk.

        Args:
            path: Directory containing saved index files.
        """
        path = Path(path)

        self.index = faiss.read_index(str(path / "index.faiss"))

        with open(path / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.chunks = [
            Chunk(
                text=m["text"],
                source=m["source"],
                chunk_id=m["chunk_id"],
                start_char=m.get("start_char", 0),
                end_char=m.get("end_char", 0),
            )
            for m in metadata
        ]

    @property
    def size(self) -> int:
        """Number of chunks in the index."""
        return self.index.ntotal
