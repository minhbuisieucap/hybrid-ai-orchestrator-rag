"""Typed values shared by the retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    page: int | None
    chunk_index: int
    text: str
    char_start: int | None = None
    char_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=str(value["chunk_id"]),
            source=str(value["source"]),
            page=int(value["page"]) if value.get("page") is not None else None,
            chunk_index=int(value["chunk_index"]),
            text=str(value["text"]),
            char_start=(
                int(value["char_start"])
                if value.get("char_start") is not None
                else None
            ),
            char_end=(
                int(value["char_end"]) if value.get("char_end") is not None else None
            ),
        )

    def location(self) -> str:
        if self.page is None:
            return self.source
        return f"{self.source}, p.{self.page}"


@dataclass(frozen=True)
class RankedChunk:
    chunk: Chunk
    score: float
    rank: int


@dataclass(frozen=True)
class FusedHit:
    chunk: Chunk
    fused_score: float
    dense_rank: int | None
    bm25_rank: int | None
    dense_score: float | None
    bm25_score: float | None
