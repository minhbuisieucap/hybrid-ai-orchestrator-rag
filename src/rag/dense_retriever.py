"""Dense retrieval over persisted, L2-normalized embeddings."""

from __future__ import annotations

import numpy as np

from src.rag.types import Chunk, RankedChunk


def dense_retrieve(
    query_embedding: np.ndarray,
    chunks: list[Chunk] | tuple[Chunk, ...],
    document_embeddings: np.ndarray,
    *,
    top_k: int,
) -> list[RankedChunk]:
    """Rank chunks by cosine similarity.

    Query and stored vectors are normalized here, so the dot product is cosine
    similarity even if a stored vector was not pre-normalized. Deterministic
    ties are broken by chunk ID.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    matrix = np.asarray(document_embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("document_embeddings must be a 2D matrix")
    if matrix.shape[0] != len(chunks):
        raise ValueError("stored chunks and embeddings do not match")
    if matrix.shape[1] != query.shape[0]:
        raise ValueError("query and document embedding dimensions do not match")
    if len(chunks) == 0:
        return []

    query_norm = float(np.linalg.norm(query))
    if query_norm:
        query = query / query_norm
    document_norms = np.linalg.norm(matrix, axis=1)
    scores = np.divide(
        matrix @ query,
        document_norms,
        out=np.zeros(matrix.shape[0], dtype=np.float32),
        where=document_norms > 0,
    )
    order = sorted(
        range(len(chunks)),
        key=lambda index: (-float(scores[index]), chunks[index].chunk_id),
    )[: min(top_k, len(chunks))]

    return [
        RankedChunk(
            chunk=chunks[index],
            score=float(scores[index]),
            rank=position + 1,
        )
        for position, index in enumerate(order)
    ]
