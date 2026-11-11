"""RAG Q&A System -- Retrieval-Augmented Generation for document Q&A."""

from rag_qa.pipeline import RAGPipeline
from rag_qa.config import RAGConfig
from rag_qa.chunker import TextChunker
from rag_qa.embedder import Embedder
from rag_qa.retriever import Retriever

__version__ = "0.1.0"
__all__ = ["RAGPipeline", "RAGConfig", "TextChunker", "Embedder", "Retriever"]
