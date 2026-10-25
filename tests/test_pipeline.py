from __future__ import annotations

import numpy as np

from src.rag.pipeline import HybridRAGPipeline


class QueryEmbedder:
    model_name = "query"

    def __init__(self, vector: list[float]) -> None:
        self.vector = np.asarray(vector, dtype=np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.vector.reshape(1, -1)


def test_pipeline_combines_dense_and_bm25_and_selects_evidence(
    tmp_path, make_settings, chunk_factory
) -> None:
    settings = make_settings(tmp_path)
    chunks = [
        chunk_factory("refund", "Refunds are available within 30 days."),
        chunk_factory("warranty", "The warranty lasts one year."),
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    pipeline = HybridRAGPipeline(
        chunks=chunks,
        embeddings=embeddings,
        embedder=QueryEmbedder([1.0, 0.0]),
        settings=settings,
    )

    result = pipeline.retrieve("refund policy")

    assert result.sufficient is True
    assert result.selected_hits
    assert result.selected_hits[0].chunk.chunk_id == "refund"
    assert result.fused_hits[0].chunk.chunk_id == "refund"


def test_pipeline_marks_weak_evidence_as_insufficient(
    tmp_path, make_settings, chunk_factory
) -> None:
    settings = make_settings(tmp_path)
    chunks = [chunk_factory("warranty", "The warranty lasts one year.")]
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    pipeline = HybridRAGPipeline(
        chunks=chunks,
        embeddings=embeddings,
        embedder=QueryEmbedder([0.0, 1.0]),
        settings=settings,
    )

    result = pipeline.retrieve("submarine propulsion")

    assert result.sufficient is False
    assert result.selected_hits == ()
