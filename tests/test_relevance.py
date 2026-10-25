from __future__ import annotations

from src.rag.pipeline import assess_evidence


def test_low_dense_and_low_bm25_reject_evidence(
    tmp_path, make_settings, chunk_factory, ranked_factory
) -> None:
    settings = make_settings(tmp_path)
    chunk = chunk_factory("c1", "unrelated text")
    dense = [ranked_factory(chunk, 0.10, 1)]
    bm25 = [ranked_factory(chunk, -1.0, 1)]

    assessment = assess_evidence(dense, bm25, "refund policy", settings)

    assert assessment.sufficient is False
    assert assessment.best_dense_score == 0.10
    assert assessment.best_bm25_score == -1.0


def test_strong_dense_score_is_sufficient(
    tmp_path, make_settings, chunk_factory, ranked_factory
) -> None:
    settings = make_settings(tmp_path)
    chunk = chunk_factory("c1", "semantically related")
    dense = [ranked_factory(chunk, 0.80, 1)]

    assessment = assess_evidence(dense, [], "question", settings)

    assert assessment.sufficient is True
    assert "dense" in assessment.reason


def test_positive_bm25_with_coverage_is_sufficient(
    tmp_path, make_settings, chunk_factory, ranked_factory
) -> None:
    settings = make_settings(tmp_path)
    chunk = chunk_factory("c1", "the refund policy applies")
    bm25 = [ranked_factory(chunk, 2.0, 1)]

    assessment = assess_evidence([], bm25, "refund policy", settings)

    assert assessment.sufficient is True
    assert assessment.best_bm25_coverage == 1.0
    assert "lexical" in assessment.reason


def test_positive_bm25_without_enough_coverage_is_not_sufficient(
    tmp_path, make_settings, chunk_factory, ranked_factory
) -> None:
    settings = make_settings(tmp_path)
    chunk = chunk_factory("c1", "refund rules")
    bm25 = [ranked_factory(chunk, 2.0, 1)]

    assessment = assess_evidence([], bm25, "refund warranty shipping", settings)

    assert assessment.sufficient is False
