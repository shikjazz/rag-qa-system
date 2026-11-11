"""Configuration for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RAGConfig:
    """Configuration for the RAG Q&A pipeline.

    Attributes:
        chunk_size: Number of characters per text chunk.
        chunk_overlap: Overlap between consecutive chunks.
        top_k: Number of relevant chunks to retrieve.
        embedding_model: Sentence-transformers model for embeddings.
        embedding_dim: Dimension of embedding vectors (auto-detected if None).
        similarity_threshold: Minimum similarity score for retrieval.
        index_path: Path to persist the FAISS index.
        max_context_length: Maximum context tokens for the LLM.
    """

    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int | None = None
    similarity_threshold: float = 0.3
    index_path: str = "./rag_index"
    max_context_length: int = 2048
    supported_extensions: list[str] = field(
        default_factory=lambda: [".txt", ".md", ".pdf"]
    )
