"""Document embedding using sentence-transformers."""

from __future__ import annotations

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment, misc]

from rag_qa.chunker import Chunk


class Embedder:
    """Generate dense vector embeddings for text chunks.

    Uses sentence-transformers models for high-quality
    semantic embeddings suitable for similarity search.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the embedder.

        Args:
            model_name: Sentence-transformers model identifier.
        """
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as numpy array.
        """
        return self.model.encode(text, normalize_embeddings=True)

    def embed_chunks(self, chunks: list[Chunk], batch_size: int = 32) -> np.ndarray:
        """Embed a list of text chunks.

        Args:
            chunks: List of Chunk objects.
            batch_size: Batch size for encoding.

        Returns:
            Matrix of embeddings (num_chunks x embedding_dim).
        """
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query.

        Args:
            query: Question or search query.

        Returns:
            Query embedding vector.
        """
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )
        return np.array(embedding, dtype=np.float32).reshape(1, -1)
