from __future__ import annotations

import pytest

from src.rag.fusion import fuse_rankings


def test_rrf_scores_duplicate_and_single_list_chunks(
    chunk_factory, ranked_factory
) -> None:
    a = chunk_factory("a", "alpha")
    b = chunk_factory("b", "beta")
    c = chunk_factory("c", "gamma")
    dense = [ranked_factory(a, 0.9, 1), ranked_factory(b, 0.8, 2)]
    bm25 = [ranked_factory(b, 4.0, 1), ranked_factory(c, 3.0, 2)]

    fused = fuse_rankings(dense, bm25, k=60, top_k=10)

    assert [hit.chunk.chunk_id for hit in fused] == ["b", "a", "c"]
    by_id = {hit.chunk.chunk_id: hit for hit in fused}
    assert by_id["a"].fused_score == pytest.approx(1 / 61)
    assert by_id["b"].fused_score == pytest.approx(1 / 62 + 1 / 61)
    assert by_id["c"].fused_score == pytest.approx(1 / 62)
    assert by_id["a"].dense_rank == 1
    assert by_id["a"].bm25_rank is None
    assert by_id["c"].dense_rank is None
    assert by_id["c"].bm25_rank == 2


def test_rrf_duplicate_in_same_list_uses_best_rank(
    chunk_factory, ranked_factory
) -> None:
    a = chunk_factory("a", "alpha")
    dense = [ranked_factory(a, 0.9, 1), ranked_factory(a, 0.1, 3)]

    fused = fuse_rankings(dense, [], k=60, top_k=5)

    assert len(fused) == 1
    assert fused[0].dense_rank == 1
    assert fused[0].fused_score == pytest.approx(1 / 61)


def test_rrf_tie_is_broken_by_chunk_id(chunk_factory, ranked_factory) -> None:
    a = chunk_factory("a", "alpha")
    b = chunk_factory("b", "beta")
    dense = [ranked_factory(a, 1.0, 1)]
    bm25 = [ranked_factory(b, 1.0, 1)]

    fused = fuse_rankings(dense, bm25, k=60, top_k=5)

    assert [hit.chunk.chunk_id for hit in fused] == ["a", "b"]


def test_rrf_rejects_invalid_parameters(chunk_factory, ranked_factory) -> None:
    a = chunk_factory("a", "alpha")
    hits = [ranked_factory(a, 1.0, 1)]

    with pytest.raises(ValueError):
        fuse_rankings(hits, [], k=0, top_k=1)
    with pytest.raises(ValueError):
        fuse_rankings(hits, [], k=60, top_k=0)
