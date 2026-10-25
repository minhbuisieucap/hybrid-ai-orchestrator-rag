"""Local sentence-transformers embedding support."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingError(RuntimeError):
    """Raised when the local embedding model cannot be loaded or used."""


class Embedder(Protocol):
    model_name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return one L2-normalized float32 row per input text."""


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        try:
            self._model = SentenceTransformer(
                self.model_name, local_files_only=True
            )
        except OSError:
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise EmbeddingError(
                    f"Could not load embedding model {self.model_name!r}. "
                    "Check the model name and network access for the first download."
                ) from exc
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        model = self._load()
        try:
            vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:
            raise EmbeddingError("Embedding inference failed") from exc

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[0] != len(texts) or matrix.ndim != 2:
            raise EmbeddingError("Embedding model returned an unexpected shape")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(
            matrix,
            norms,
            out=np.zeros_like(matrix),
            where=norms > 0,
        )
