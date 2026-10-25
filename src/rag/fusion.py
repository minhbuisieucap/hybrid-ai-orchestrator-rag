"""Reciprocal Rank Fusion for dense and lexical result lists."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Sequence

from src.rag.types import Chunk, FusedHit, RankedChunk


@dataclass
class _Accumulator:
    chunk: Chunk
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    bm25_score: float | None = None


def _best_rank(current: int | None, candidate: int) -> int:
    return candidate if current is None else min(current, candidate)


def _best_score(current: float | None, candidate: float) -> float:
    return candidate if current is None else max(current, candidate)


def fuse_rankings(
    dense_hits: Sequence[RankedChunk],
    bm25_hits: Sequence[RankedChunk],
    *,
    k: int = 60,
    top_k: int = 5,
) -> list[FusedHit]:
    """Fuse two independently ranked lists using one-based RRF ranks.

    Raw dense similarity and BM25 scores are retained for debugging only; they
    never contribute directly to the fused score.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    accumulated: dict[str, _Accumulator] = {}

    def add(hits: Sequence[RankedChunk], modality: str) -> None:
        for fallback_rank, hit in enumerate(hits, start=1):
            rank = hit.rank if hit.rank > 0 else fallback_rank
            item = accumulated.get(hit.chunk.chunk_id)
            if item is None:
                item = _Accumulator(chunk=hit.chunk)
                accumulated[hit.chunk.chunk_id] = item
            if modality == "dense":
                item.dense_rank = _best_rank(item.dense_rank, rank)
                item.dense_score = _best_score(item.dense_score, hit.score)
            else:
                item.bm25_rank = _best_rank(item.bm25_rank, rank)
                item.bm25_score = _best_score(item.bm25_score, hit.score)

    add(dense_hits, "dense")
    add(bm25_hits, "bm25")

    fused: list[FusedHit] = []
    for item in accumulated.values():
        score = 0.0
        if item.dense_rank is not None:
            score += 1.0 / (k + item.dense_rank)
        if item.bm25_rank is not None:
            score += 1.0 / (k + item.bm25_rank)
        fused.append(
            FusedHit(
                chunk=item.chunk,
                fused_score=score,
                dense_rank=item.dense_rank,
                bm25_rank=item.bm25_rank,
                dense_score=item.dense_score,
                bm25_score=item.bm25_score,
            )
        )

    fused.sort(
        key=lambda hit: (
            -hit.fused_score,
            min(
                hit.dense_rank if hit.dense_rank is not None else inf,
                hit.bm25_rank if hit.bm25_rank is not None else inf,
            ),
            hit.chunk.chunk_id,
        )
    )
    return fused[: min(top_k, len(fused))]
