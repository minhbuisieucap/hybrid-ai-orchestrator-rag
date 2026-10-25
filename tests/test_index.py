from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from src.rag.index import IndexStore


class CountingEmbedder:
    def __init__(self, model_name: str = "fake-embedder") -> None:
        self.model_name = model_name
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = np.frombuffer(digest[:4], dtype=np.uint8).astype(np.float32)
            norm = float(np.linalg.norm(vector))
            rows.append(vector / norm)
        return np.vstack(rows).astype(np.float32)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_saves_index_and_manifest(tmp_path, make_settings) -> None:
    settings = make_settings(tmp_path)
    _write(settings.data_dir / "a.md", "Refunds are available within 30 days.")
    _write(settings.data_dir / "sub" / "b.txt", "Warranty lasts one year.")
    embedder = CountingEmbedder()
    store = IndexStore(settings)

    report = store.build(embedder)

    assert report.chunks == 2
    assert set(report.added) == {"a.md", "sub/b.txt"}
    assert store.exists()
    loaded = store.load()
    assert len(loaded.chunks) == 2
    assert loaded.embeddings.shape == (2, 4)
    assert loaded.manifest["embedding_model"] == "fake-embedder"


def test_unchanged_documents_are_not_reembedded(tmp_path, make_settings) -> None:
    settings = make_settings(tmp_path)
    _write(settings.data_dir / "a.md", "Refunds are available within 30 days.")
    embedder = CountingEmbedder()
    store = IndexStore(settings)
    store.build(embedder)
    call_count = len(embedder.calls)

    report = store.build(embedder)

    assert report.updated == []
    assert len(embedder.calls) == call_count


def test_changed_document_is_reembedded_and_removed_document_is_dropped(
    tmp_path, make_settings
) -> None:
    settings = make_settings(tmp_path)
    _write(settings.data_dir / "a.md", "Refunds are available within 30 days.")
    _write(settings.data_dir / "b.md", "Warranty lasts one year.")
    embedder = CountingEmbedder()
    store = IndexStore(settings)
    store.build(embedder)
    call_count = len(embedder.calls)

    _write(settings.data_dir / "a.md", "Refunds are available within 14 days.")
    report = store.build(embedder)

    assert report.updated == ["a.md"]
    assert len(embedder.calls) == call_count + 1
    assert "b.md" in report.unchanged
    loaded = store.load()
    assert any("14 days" in chunk.text for chunk in loaded.chunks)

    (settings.data_dir / "b.md").unlink()
    report = store.build(embedder)
    assert report.removed == ["b.md"]
    assert all(chunk.source != "b.md" for chunk in store.load().chunks)


def test_force_rebuild_reembeds_everything(tmp_path, make_settings) -> None:
    settings = make_settings(tmp_path)
    _write(settings.data_dir / "a.md", "Refunds are available within 30 days.")
    embedder = CountingEmbedder()
    store = IndexStore(settings)
    store.build(embedder)
    call_count = len(embedder.calls)

    report = store.build(embedder, force=True)

    assert report.updated == ["a.md"]
    assert len(embedder.calls) == call_count + 1


def test_empty_corpus_clears_existing_index(tmp_path, make_settings) -> None:
    settings = make_settings(tmp_path)
    doc = settings.data_dir / "a.md"
    _write(doc, "Refunds are available within 30 days.")
    store = IndexStore(settings)
    store.build(CountingEmbedder())
    doc.unlink()

    report = store.build(CountingEmbedder())

    assert report.chunks == 0
    assert not store.exists()


def test_embedding_model_change_rebuilds_embeddings(tmp_path, make_settings) -> None:
    settings = make_settings(tmp_path)
    _write(settings.data_dir / "a.md", "Refunds are available within 30 days.")
    store = IndexStore(settings)
    first = CountingEmbedder("model-a")
    store.build(first)

    second = CountingEmbedder("model-b")
    report = store.build(second)

    assert report.updated == ["a.md"]
    assert second.calls
    assert store.load().manifest["embedding_model"] == "model-b"
