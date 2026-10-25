"""Evidence formatting and citation validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from src.rag.types import Chunk

_CITATION_RE = re.compile(r"\[[Ss](\d+)\]")


@dataclass(frozen=True)
class SourceReference:
    citation_id: str
    chunk: Chunk


@dataclass(frozen=True)
class CitationValidation:
    clean_answer: str
    cited_ids: tuple[str, ...]
    invalid_ids: tuple[str, ...]


def build_source_map(chunks: Sequence[Chunk]) -> dict[str, SourceReference]:
    return {
        f"S{index}": SourceReference(citation_id=f"S{index}", chunk=chunk)
        for index, chunk in enumerate(chunks, start=1)
    }


def format_evidence(
    chunks: Sequence[Chunk],
    *,
    max_chars: int,
) -> tuple[str, dict[str, SourceReference]]:
    blocks: list[str] = []
    included: list[Chunk] = []
    used = 0
    for index, chunk in enumerate(chunks, start=1):
        location = chunk.location()
        block = (
            f"<<<EVIDENCE [S{index}] location={location!r} "
            f"chunk_id={chunk.chunk_id!r}>>>\n"
            f"{chunk.text}\n<<<END EVIDENCE [S{index}]>>>"
        )
        if included and used + len(block) > max_chars:
            break
        blocks.append(block)
        included.append(chunk)
        used += len(block)
    context = "\n\n".join(blocks)
    return context, build_source_map(included)


def validate_citations(
    answer: str,
    sources: dict[str, SourceReference],
) -> CitationValidation:
    cited: list[str] = []
    invalid: list[str] = []

    def replace(match: re.Match[str]) -> str:
        citation_id = f"S{match.group(1)}"
        if citation_id in sources:
            if citation_id not in cited:
                cited.append(citation_id)
            return f"[{citation_id}]"
        if citation_id not in invalid:
            invalid.append(citation_id)
        return ""

    clean_answer = _CITATION_RE.sub(replace, answer)
    return CitationValidation(
        clean_answer=clean_answer.strip(),
        cited_ids=tuple(cited),
        invalid_ids=tuple(invalid),
    )


def format_sources(
    sources: dict[str, SourceReference],
    *,
    cited_ids: Iterable[str],
) -> str:
    cited = set(cited_ids)
    lines = ["Sources:"]
    for citation_id, source in sources.items():
        status = "cited" if citation_id in cited else "retrieved"
        lines.append(f"  [{citation_id}] {source.chunk.location()} ({status})")
    return "\n".join(lines)
