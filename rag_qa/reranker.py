"""
Re-ranking Module for RAG Pipeline.
Implements cross-encoder style re-ranking to improve retrieval quality.
Provides multiple re-ranking strategies: keyword overlap, BM25, and
semantic similarity-based approaches.
"""
import math
import re
from collections import Counter
from typing import Optional


def tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric."""
    return re.findall(r'\w+', text.lower())


def compute_bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    avg_doc_len: float,
    doc_freq: dict[str, int],
    n_docs: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Compute BM25 score for a single document against a query."""
    score = 0.0
    doc_len = len(doc_tokens)
    doc_counter = Counter(doc_tokens)

    for term in query_tokens:
        if term not in doc_counter:
            continue
        tf = doc_counter[term]
        df = doc_freq.get(term, 0)
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
        score += idf * tf_norm

    return score


class KeywordReranker:
    """Re-rank documents based on keyword overlap with query."""

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """Re-rank documents by keyword overlap score."""
        query_tokens = set(tokenize(query))
        scored = []

        for doc in documents:
            doc_tokens = set(tokenize(doc.get("text", "")))
            overlap = len(query_tokens & doc_tokens)
            coverage = overlap / max(len(query_tokens), 1)
            scored.append({**doc, "rerank_score": coverage})

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k] if top_k else scored


class BM25Reranker:
    """Re-rank documents using BM25 scoring."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """Re-rank documents using BM25."""
        query_tokens = tokenize(query)
        all_doc_tokens = [tokenize(d.get("text", "")) for d in documents]

        # Compute document frequencies
        doc_freq = Counter()
        for tokens in all_doc_tokens:
            for term in set(tokens):
                doc_freq[term] += 1

        avg_doc_len = sum(len(t) for t in all_doc_tokens) / max(len(all_doc_tokens), 1)
        n_docs = len(documents)

        scored = []
        for doc, tokens in zip(documents, all_doc_tokens):
            score = compute_bm25_score(
                query_tokens, tokens, avg_doc_len, doc_freq, n_docs,
                self.k1, self.b,
            )
            scored.append({**doc, "rerank_score": score})

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k] if top_k else scored


class RecipRankFusion:
    """Reciprocal Rank Fusion for combining multiple ranking lists."""

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        rankings: list[list[dict]],
        id_key: str = "id",
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """Fuse multiple rankings using reciprocal rank fusion."""
        scores = {}
        docs_by_id = {}

        for ranking in rankings:
            for rank, doc in enumerate(ranking):
                doc_id = doc.get(id_key, str(rank))
                rrf_score = 1.0 / (self.k + rank + 1)
                scores[doc_id] = scores.get(doc_id, 0) + rrf_score
                docs_by_id[doc_id] = doc

        fused = []
        for doc_id, score in sorted(scores.items(), key=lambda x: -x[1]):
            fused.append({**docs_by_id[doc_id], "fusion_score": score})

        return fused[:top_k] if top_k else fused


if __name__ == "__main__":
    # Demo
    query = "What is machine learning used for in NLP?"
    documents = [
        {"id": "1", "text": "Machine learning powers many NLP applications like translation and summarization."},
        {"id": "2", "text": "Deep learning has revolutionized computer vision tasks."},
        {"id": "3", "text": "NLP uses machine learning for text classification, named entity recognition, and more."},
        {"id": "4", "text": "Python is a popular programming language for data science."},
    ]

    print("=== Keyword Reranker ===")
    kr = KeywordReranker()
    for doc in kr.rerank(query, documents):
        print(f"  [{doc['id']}] score={doc['rerank_score']:.3f}: {doc['text'][:60]}...")

    print("\n=== BM25 Reranker ===")
    bm25 = BM25Reranker()
    for doc in bm25.rerank(query, documents):
        print(f"  [{doc['id']}] score={doc['rerank_score']:.3f}: {doc['text'][:60]}...")
"""
