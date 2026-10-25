from __future__ import annotations

import numpy as np
import pytest

from src.rag.dense_retriever import dense_retrieve


def _chunks(chunk_factory):
    return [
        chunk_factory("c1", "first"),
        chunk_factory("c2", "second"),
        chunk_factory("c3", "third"),
    ]


def test_dense_retrieval_returns_ranked_hits(chunk_factory) -> None:
    chunks = _chunks(chunk_factory)
    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.70710678, 0.70710678]], dtype=np.float32
    )

    hits = dense_retrieve(
        np.array([1.0, 0.0], dtype=np.float32), chunks, embeddings, top_k=2
    )

    assert [hit.chunk.chunk_id for hit in hits] == ["c1", "c3"]
    assert [hit.rank for hit in hits] == [1, 2]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(0.70710678)


def test_dense_retrieval_is_deterministic_on_ties(chunk_factory) -> None:
    chunks = [
        chunk_factory("b", "b"),
        chunk_factory("a", "a"),
    ]
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    hits = dense_retrieve(np.array([1.0, 0.0]), chunks, embeddings, top_k=2)

    assert [hit.chunk.chunk_id for hit in hits] == ["a", "b"]


def test_dense_retrieval_computes_cosine_for_unnormalized_rows(
    chunk_factory,
) -> None:
    chunks = [chunk_factory("c1", "first")]
    embeddings = np.array([[10.0, 0.0]], dtype=np.float32)

    hits = dense_retrieve(
        np.array([2.0, 0.0], dtype=np.float32), chunks, embeddings, top_k=1
    )

    assert hits[0].score == pytest.approx(1.0)


def test_dense_retrieval_validates_shapes(chunk_factory) -> None:
    chunks = _chunks(chunk_factory)

    with pytest.raises(ValueError):
        dense_retrieve(
            np.array([1.0, 0.0]),
            chunks,
            np.ones((2, 2), dtype=np.float32),
            top_k=1,
        )

    with pytest.raises(ValueError):
        dense_retrieve(
            np.array([1.0, 0.0, 0.0]),
            chunks,
            np.ones((3, 2), dtype=np.float32),
            top_k=1,
        )
