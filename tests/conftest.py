from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from config import Settings
from src.rag.types import Chunk, RankedChunk


class FakeEmbedder:
    """Deterministic 4-dimensional embedder for tests; no model download."""

    model_name = "fake-embedder"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = np.array(
                [digest[0], digest[1], digest[2], digest[3]], dtype=np.float32
            )
            norm = float(np.linalg.norm(vector))
            rows.append(vector / norm if norm else vector)
        return np.vstack(rows).astype(np.float32)


class FixedQueryEmbedder:
    model_name = "fixed-query"

    def __init__(self, vector: list[float]) -> None:
        self.vector = np.asarray(vector, dtype=np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        assert len(texts) == 1
        return self.vector.reshape(1, -1)


def _build_settings(tmp_path: Path, **overrides) -> Settings:
    data_dir = tmp_path / "data"
    storage_dir = tmp_path / "storage"
    data_dir.mkdir(exist_ok=True)
    storage_dir.mkdir(exist_ok=True)
    values = dict(
        project_root=tmp_path,
        data_dir=data_dir,
        storage_dir=storage_dir,
        embedding_model="fake-embedder",
        chunk_size=100,
        chunk_overlap=20,
        dense_top_k=3,
        bm25_top_k=3,
        fused_top_k=3,
        rrf_k=60,
        dense_min_similarity=0.30,
        bm25_min_score=0.0,
        bm25_min_coverage=0.67,
        max_context_chars=2000,
        local_only=True,
        allow_cloud_fallback=False,
        allow_cloud_rag=False,
        default_provider="ollama",
        provider_priority=("ollama", "openai", "gemini"),
        ollama_host="http://localhost:11434",
        ollama_model="qwen3:1.7b",
        openai_model="",
        gemini_model="",
        request_timeout_seconds=5.0,
        llm_temperature=0.0,
        history_messages=4,
        embedding_batch_size=8,
    )
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def make_settings():
    return _build_settings


@pytest.fixture
def chunk_factory():
    def make(
        chunk_id: str,
        text: str,
        source: str = "a.md",
        page: int | None = None,
        chunk_index: int = 0,
        char_start: int | None = None,
        char_end: int | None = None,
    ) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            source=source,
            page=page,
            chunk_index=chunk_index,
            text=text,
            char_start=char_start,
            char_end=char_end,
        )

    return make


@pytest.fixture
def ranked_factory():
    def make(chunk: Chunk, score: float, rank: int) -> RankedChunk:
        return RankedChunk(chunk=chunk, score=score, rank=rank)

    return make
