"""Hybrid retrieval: dense search, BM25 search, RRF, and evidence gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from config import Settings
from src.rag.bm25_retriever import BM25Retriever, token_coverage
from src.rag.dense_retriever import dense_retrieve
from src.rag.embeddings import Embedder
from src.rag.fusion import fuse_rankings
from src.rag.types import Chunk, FusedHit, RankedChunk


@dataclass(frozen=True)
class EvidenceAssessment:
    sufficient: bool
    reason: str
    best_dense_score: float | None
    best_bm25_score: float | None
    best_bm25_coverage: float


@dataclass(frozen=True)
class HybridRetrievalResult:
    selected_hits: tuple[FusedHit, ...]
    fused_hits: tuple[FusedHit, ...]
    dense_hits: tuple[RankedChunk, ...]
    bm25_hits: tuple[RankedChunk, ...]
    sufficient: bool
    reason: str
    best_dense_score: float | None
    best_bm25_score: float | None
    best_bm25_coverage: float


def assess_evidence(
    dense_hits: Sequence[RankedChunk],
    bm25_hits: Sequence[RankedChunk],
    query: str,
    settings: Settings,
) -> EvidenceAssessment:
    best_dense = max((hit.score for hit in dense_hits), default=None)
    best_bm25_hit = max(bm25_hits, key=lambda hit: hit.score, default=None)
    best_bm25 = best_bm25_hit.score if best_bm25_hit is not None else None
    best_coverage = (
        token_coverage(query, best_bm25_hit.chunk.text)
        if best_bm25_hit is not None
        else 0.0
    )

    dense_ok = (
        best_dense is not None and best_dense >= settings.dense_min_similarity
    )
    bm25_ok = (
        best_bm25 is not None
        and best_bm25 > 0
        and best_bm25 >= settings.bm25_min_score
        and best_coverage >= settings.bm25_min_coverage
    )

    if dense_ok and bm25_ok:
        reason = (
            "dense and lexical evidence passed their independent thresholds"
        )
    elif dense_ok:
        reason = (
            f"dense similarity {best_dense:.3f} >= "
            f"{settings.dense_min_similarity:.3f}"
        )
    elif bm25_ok:
        reason = (
            f"lexical BM25 score {best_bm25:.3f} with query coverage "
            f"{best_coverage:.2f}"
        )
    else:
        reason = "no retrieval method passed its independent relevance threshold"

    return EvidenceAssessment(
        sufficient=dense_ok or bm25_ok,
        reason=reason,
        best_dense_score=best_dense,
        best_bm25_score=best_bm25,
        best_bm25_coverage=best_coverage,
    )


class HybridRAGPipeline:
    def __init__(
        self,
        *,
        chunks: Sequence[Chunk],
        embeddings: np.ndarray,
        embedder: Embedder,
        settings: Settings,
    ) -> None:
        self.chunks = list(chunks)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.embedder = embedder
        self.settings = settings
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(
            self.chunks
        ):
            raise ValueError("chunks and embeddings do not match")
        self.bm25 = BM25Retriever(self.chunks)

    def retrieve(self, query: str) -> HybridRetrievalResult:
        if not self.chunks:
            assessment = EvidenceAssessment(
                sufficient=False,
                reason="the local index is empty",
                best_dense_score=None,
                best_bm25_score=None,
                best_bm25_coverage=0.0,
            )
            return HybridRetrievalResult(
                selected_hits=(),
                fused_hits=(),
                dense_hits=(),
                bm25_hits=(),
                sufficient=False,
                reason=assessment.reason,
                best_dense_score=None,
                best_bm25_score=None,
                best_bm25_coverage=0.0,
            )

        query_matrix = self.embedder.encode([query])
        if query_matrix.shape[0] != 1:
            raise ValueError("embedder must return one row for one query")
        dense_hits = dense_retrieve(
            query_matrix[0],
            self.chunks,
            self.embeddings,
            top_k=self.settings.dense_top_k,
        )
        bm25_hits = self.bm25.search(query, top_k=self.settings.bm25_top_k)
        fused = fuse_rankings(
            dense_hits,
            bm25_hits,
            k=self.settings.rrf_k,
            top_k=self.settings.fused_top_k,
        )
        assessment = assess_evidence(
            dense_hits, bm25_hits, query, self.settings
        )

        selected: list[FusedHit] = []
        if assessment.sufficient:
            used = 0
            for hit in fused:
                dense_ok = (
                    hit.dense_score is not None
                    and hit.dense_score >= self.settings.dense_min_similarity
                )
                bm25_ok = (
                    hit.bm25_score is not None
                    and hit.bm25_score > 0
                    and hit.bm25_score >= self.settings.bm25_min_score
                    and token_coverage(query, hit.chunk.text)
                    >= self.settings.bm25_min_coverage
                )
                if not (dense_ok or bm25_ok):
                    continue
                if (
                    selected
                    and used + len(hit.chunk.text)
                    > self.settings.max_context_chars
                ):
                    break
                selected.append(hit)
                used += len(hit.chunk.text)

        return HybridRetrievalResult(
            selected_hits=tuple(selected),
            fused_hits=tuple(fused),
            dense_hits=tuple(dense_hits),
            bm25_hits=tuple(bm25_hits),
            sufficient=assessment.sufficient,
            reason=assessment.reason,
            best_dense_score=assessment.best_dense_score,
            best_bm25_score=assessment.best_bm25_score,
            best_bm25_coverage=assessment.best_bm25_coverage,
        )
