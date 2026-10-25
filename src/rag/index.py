"""Persistent local index with content-hash based incremental rebuilds."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from config import Settings
from src.rag.embeddings import Embedder
from src.rag.ingestion import (
    IngestionIssue,
    extract_document,
    find_documents,
    sha256_file,
)
from src.rag.types import Chunk

INDEX_VERSION = 1


class IndexStorageError(RuntimeError):
    """Raised when a local index cannot be loaded safely."""


@dataclass(frozen=True)
class IndexData:
    chunks: list[Chunk]
    embeddings: np.ndarray
    manifest: dict[str, Any]


@dataclass
class BuildReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    chunks: int = 0
    errors: list[IngestionIssue] = field(default_factory=list)


class IndexStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def exists(self) -> bool:
        return (
            self.settings.chunks_file.is_file()
            and self.settings.embeddings_file.is_file()
            and self.settings.manifest_file.is_file()
        )

    def clear(self) -> None:
        for path in (
            self.settings.chunks_file,
            self.settings.embeddings_file,
            self.settings.manifest_file,
        ):
            if path.exists():
                path.unlink()

    def _save(self, chunks: list[Chunk], embeddings: np.ndarray, manifest: dict) -> None:
        self.settings.ensure_directories()
        chunks_temp = self.settings.chunks_file.with_suffix(".json.tmp")
        embeddings_temp = self.settings.embeddings_file.with_suffix(".npy.tmp")
        manifest_temp = self.settings.manifest_file.with_suffix(".json.tmp")

        chunks_temp.write_text(
            json.dumps(
                [chunk.to_dict() for chunk in chunks],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        with embeddings_temp.open("wb") as handle:
            np.save(handle, embeddings)
        manifest_temp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        chunks_temp.replace(self.settings.chunks_file)
        embeddings_temp.replace(self.settings.embeddings_file)
        manifest_temp.replace(self.settings.manifest_file)

    def load(self) -> IndexData:
        if not self.exists():
            raise IndexStorageError("No local index found. Run --ingest to build it.")
        try:
            manifest = json.loads(
                self.settings.manifest_file.read_text(encoding="utf-8")
            )
            raw_chunks = json.loads(
                self.settings.chunks_file.read_text(encoding="utf-8")
            )
            embeddings = np.load(
                self.settings.embeddings_file, allow_pickle=False
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise IndexStorageError(f"Could not load the local index: {exc}") from exc

        if manifest.get("version") != INDEX_VERSION:
            raise IndexStorageError("The local index version is unsupported. Rebuild it.")
        if not isinstance(manifest.get("documents"), dict):
            raise IndexStorageError("The local index manifest is malformed.")
        if not isinstance(raw_chunks, list) or not raw_chunks:
            raise IndexStorageError("The local index is empty. Run --ingest.")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(raw_chunks):
            raise IndexStorageError("The index chunks and embeddings do not match.")
        if manifest.get("embedding_dimension") != embeddings.shape[1]:
            raise IndexStorageError("The index embedding dimension is inconsistent.")
        try:
            chunks = [Chunk.from_dict(value) for value in raw_chunks]
        except (KeyError, TypeError, ValueError) as exc:
            raise IndexStorageError("The local index metadata is malformed.") from exc
        return IndexData(chunks=chunks, embeddings=embeddings, manifest=manifest)

    def _load_for_reuse(
        self, embedder: Embedder
    ) -> tuple[IndexData | None, set[str]]:
        if not self.exists():
            return None, set()
        try:
            old = self.load()
        except IndexStorageError:
            return None, set()
        old_sources = set(old.manifest.get("documents", {}))
        compatible = (
            old.manifest.get("embedding_model") == embedder.model_name
            and old.manifest.get("chunk_size") == self.settings.chunk_size
            and old.manifest.get("chunk_overlap") == self.settings.chunk_overlap
        )
        return (old if compatible else None), old_sources

    def build(self, embedder: Embedder, *, force: bool = False) -> BuildReport:
        self.settings.ensure_directories()
        old, old_sources = self._load_for_reuse(embedder)
        old_manifest_docs = old.manifest.get("documents", {}) if old else {}
        old_chunks_by_id = {
            chunk.chunk_id: chunk for chunk in old.chunks
        } if old else {}
        old_rows_by_id = {
            chunk.chunk_id: old.embeddings[index]
            for index, chunk in enumerate(old.chunks)
        } if old else {}

        documents = find_documents(self.settings.data_dir)
        present_sources = {
            path.relative_to(self.settings.data_dir).as_posix() for path in documents
        }
        report = BuildReport(
            removed=sorted(old_sources - present_sources),
        )
        reused_chunks: list[Chunk] = []
        reused_rows: list[np.ndarray] = []
        pending_chunks: list[Chunk] = []
        document_metadata: dict[str, dict[str, Any]] = {}

        for path in documents:
            source = path.relative_to(self.settings.data_dir).as_posix()
            try:
                content_hash = sha256_file(path)
            except OSError as exc:
                report.errors.append(
                    IngestionIssue(source, f"could not read file: {exc}")
                )
                continue

            previous = old_manifest_docs.get(source)
            can_reuse = (
                not force
                and old is not None
                and isinstance(previous, dict)
                and previous.get("sha256") == content_hash
                and all(
                    chunk_id in old_chunks_by_id
                    for chunk_id in previous.get("chunk_ids", [])
                )
            )
            if can_reuse:
                chunk_ids = previous.get("chunk_ids", [])
                for chunk_id in chunk_ids:
                    reused_chunks.append(old_chunks_by_id[chunk_id])
                    reused_rows.append(old_rows_by_id[chunk_id])
                document_metadata[source] = previous
                report.unchanged.append(source)
                continue

            chunks, issues = extract_document(
                path,
                relative_source=source,
                chunk_size=self.settings.chunk_size,
                overlap=self.settings.chunk_overlap,
            )
            report.errors.extend(issues)
            if not chunks:
                continue
            pending_chunks.extend(chunks)
            document_metadata[source] = {
                "sha256": content_hash,
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "chunk_ids": [chunk.chunk_id for chunk in chunks],
            }
            if source in old_sources:
                report.updated.append(source)
            else:
                report.added.append(source)

        pending_rows: list[np.ndarray] = []
        if pending_chunks:
            matrix = embedder.encode([chunk.text for chunk in pending_chunks])
            if matrix.ndim != 2 or matrix.shape[0] != len(pending_chunks):
                raise IndexStorageError("Embedding model returned an unexpected shape.")
            pending_rows = [matrix[index] for index in range(matrix.shape[0])]

        combined = list(zip(reused_chunks + pending_chunks, reused_rows + pending_rows))
        combined.sort(
            key=lambda item: (
                item[0].source,
                item[0].page or 0,
                item[0].chunk_index,
            )
        )
        chunks = [item[0] for item in combined]
        rows = [item[1] for item in combined]

        if not chunks:
            self.clear()
            report.chunks = 0
            return report

        embeddings = np.vstack(rows).astype(np.float32)
        manifest = {
            "version": INDEX_VERSION,
            "embedding_model": embedder.model_name,
            "embedding_dimension": int(embeddings.shape[1]),
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "document_count": len(document_metadata),
            "chunk_count": len(chunks),
            "documents": document_metadata,
        }
        self._save(chunks, embeddings, manifest)
        report.chunks = len(chunks)
        return report
