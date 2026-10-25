"""Document discovery, text extraction, and safe ingestion errors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from src.rag.chunker import chunk_text
from src.rag.types import Chunk

SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".pdf"})


@dataclass(frozen=True)
class IngestionIssue:
    source: str
    message: str


@dataclass(frozen=True)
class IngestionResult:
    chunks: list[Chunk]
    issues: list[IngestionIssue]


def find_documents(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_error(exc: Exception) -> str:
    message = " ".join(str(exc).split()) or exc.__class__.__name__
    return f"{exc.__class__.__name__}: {message[:200]}"


def _extract_parts(path: Path) -> list[tuple[int | None, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [
            (index + 1, page.extract_text() or "")
            for index, page in enumerate(reader.pages)
        ]
    text = path.read_text(encoding="utf-8-sig")
    return [(None, text)]


def extract_document(
    path: Path,
    *,
    relative_source: str,
    chunk_size: int,
    overlap: int,
) -> tuple[list[Chunk], list[IngestionIssue]]:
    chunks: list[Chunk] = []
    issues: list[IngestionIssue] = []
    try:
        parts = _extract_parts(path)
    except Exception as exc:
        return [], [IngestionIssue(relative_source, _clean_error(exc))]

    for page, part_text in parts:
        if not part_text.strip():
            continue
        page_chunks = chunk_text(
            part_text,
            source=relative_source,
            page=page,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        chunks.extend(page_chunks)

    if not chunks:
        issues.append(IngestionIssue(relative_source, "no extractable text"))
    return chunks, issues


def ingest_directory(
    data_dir: Path,
    *,
    chunk_size: int,
    overlap: int,
) -> IngestionResult:
    chunks: list[Chunk] = []
    issues: list[IngestionIssue] = []
    for path in find_documents(data_dir):
        relative_source = path.relative_to(data_dir).as_posix()
        document_chunks, document_issues = extract_document(
            path,
            relative_source=relative_source,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        chunks.extend(document_chunks)
        issues.extend(document_issues)
    return IngestionResult(chunks=chunks, issues=issues)
