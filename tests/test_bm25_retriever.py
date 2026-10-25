from __future__ import annotations

from src.rag.bm25_retriever import BM25Retriever, tokenize


def test_tokenizer_is_case_insensitive_and_unicode_aware() -> None:
    assert tokenize("Refund POLICY, hoàn tiền!") == ["refund", "policy", "hoàn", "tiền"]


def test_bm25_ranks_lexically_relevant_chunk_first(chunk_factory) -> None:
    chunks = [
        chunk_factory("refund", "Refunds are available within 30 days."),
        chunk_factory("warranty", "The hardware warranty lasts one year."),
        chunk_factory("shipping", "Shipping takes five business days."),
    ]
    retriever = BM25Retriever(chunks)

    hits = retriever.search("refund policy", top_k=2)

    assert hits
    assert hits[0].chunk.chunk_id == "refund"
    assert hits[0].rank == 1
    assert hits[0].score > 0


def test_bm25_returns_empty_for_empty_query(chunk_factory) -> None:
    retriever = BM25Retriever([chunk_factory("c1", "some text")])

    assert retriever.search("", top_k=3) == []


def test_bm25_handles_corpus_with_no_content_tokens(chunk_factory) -> None:
    retriever = BM25Retriever([chunk_factory("c1", "the and of")])

    assert retriever.search("refund", top_k=3) == []


def test_tokenizer_does_not_strip_is_or_us_endings() -> None:
    assert tokenize("analysis status basis") == ["analysis", "status", "basis"]


def test_bm25_query_coverage_uses_consistent_tokens(chunk_factory) -> None:
    chunk = chunk_factory("c1", "refund and return policy")
    retriever = BM25Retriever([chunk])

    assert retriever.query_coverage("refund policy", chunk) == 1.0
    assert retriever.query_coverage("warranty policy", chunk) == 0.5
