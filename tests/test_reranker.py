"""Tests for re-ranking module."""
import pytest
from rag_qa.reranker import (
    KeywordReranker, BM25Reranker, RecipRankFusion, tokenize,
)


SAMPLE_DOCS = [
    {"id": "1", "text": "Machine learning powers NLP applications"},
    {"id": "2", "text": "Deep learning for computer vision"},
    {"id": "3", "text": "NLP text classification with machine learning"},
    {"id": "4", "text": "Python programming for data science"},
]


class TestTokenize:
    def test_basic(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_punctuation(self):
        tokens = tokenize("It's a test, right?")
        assert "it" in tokens
        assert "test" in tokens


class TestKeywordReranker:
    def test_rerank_order(self):
        kr = KeywordReranker()
        results = kr.rerank("machine learning NLP", SAMPLE_DOCS)
        # Docs mentioning all query terms should rank higher
        assert results[0]["id"] in ("1", "3")

    def test_top_k(self):
        kr = KeywordReranker()
        results = kr.rerank("machine learning", SAMPLE_DOCS, top_k=2)
        assert len(results) == 2

    def test_empty_query(self):
        kr = KeywordReranker()
        results = kr.rerank("", SAMPLE_DOCS)
        assert len(results) == len(SAMPLE_DOCS)


class TestBM25Reranker:
    def test_rerank_order(self):
        bm25 = BM25Reranker()
        results = bm25.rerank("machine learning NLP", SAMPLE_DOCS)
        top_ids = [r["id"] for r in results[:2]]
        assert "1" in top_ids or "3" in top_ids

    def test_top_k(self):
        bm25 = BM25Reranker()
        results = bm25.rerank("NLP", SAMPLE_DOCS, top_k=1)
        assert len(results) == 1

    def test_scores_positive(self):
        bm25 = BM25Reranker()
        results = bm25.rerank("machine", SAMPLE_DOCS)
        for r in results:
            assert r["rerank_score"] >= 0


class TestRecipRankFusion:
    def test_fusion(self):
        ranking1 = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        ranking2 = [{"id": "b"}, {"id": "c"}, {"id": "a"}]
        rrf = RecipRankFusion()
        fused = rrf.fuse([ranking1, ranking2])
        # "b" appears high in both, should be top
        assert fused[0]["id"] == "b"

    def test_top_k(self):
        ranking1 = [{"id": "a"}, {"id": "b"}]
        ranking2 = [{"id": "b"}, {"id": "a"}]
        rrf = RecipRankFusion()
        fused = rrf.fuse([ranking1, ranking2], top_k=1)
        assert len(fused) == 1
