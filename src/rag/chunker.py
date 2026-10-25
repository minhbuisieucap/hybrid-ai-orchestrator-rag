"""Character-based chunking with overlap and stable identifiers."""

from __future__ import annotations

from src.rag.types import Chunk


def chunk_text(
    text: str,
    *,
    source: str,
    chunk_size: int,
    overlap: int,
    page: int | None = None,
) -> list[Chunk]:
    """Split text into fixed-size characters windows.

    Chunks are deliberately simple and deterministic. PDF callers pass one page
    at a time so page metadata is never guessed.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    chunk_index = 0
    for start in range(0, len(normalized), step):
        end = min(start + chunk_size, len(normalized))
        chunk_text_value = normalized[start:end].strip()
        if chunk_text_value:
            chunks.append(
                Chunk(
                    chunk_id=f"{source}#p{page or 0}#c{chunk_index}",
                    source=source,
                    page=page,
                    chunk_index=chunk_index,
                    text=chunk_text_value,
                    char_start=start,
                    char_end=end,
                )
            )
            chunk_index += 1
        if start + chunk_size >= len(normalized):
            break
    return chunks
