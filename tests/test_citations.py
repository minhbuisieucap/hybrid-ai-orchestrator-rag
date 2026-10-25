from __future__ import annotations

from src.rag.citations import (
    build_source_map,
    format_evidence,
    format_sources,
    validate_citations,
)


def test_source_ids_are_stable_and_one_based(chunk_factory) -> None:
    chunks = [
        chunk_factory("a", "alpha", source="a.md"),
        chunk_factory("b", "beta", source="b.pdf", page=4),
    ]

    sources = build_source_map(chunks)

    assert list(sources) == ["S1", "S2"]
    assert sources["S2"].chunk.page == 4


def test_evidence_contains_source_labels_and_respects_budget(chunk_factory) -> None:
    chunks = [
        chunk_factory("a", "alpha text", source="a.md"),
        chunk_factory("b", "beta text", source="b.pdf", page=4),
    ]

    context, sources = format_evidence(chunks, max_chars=1000)

    assert "[S1]" in context
    assert "a.md" in context
    assert "[S2]" in context
    assert "p.4" in context
    assert len(sources) == 2


def test_validate_citations_keeps_valid_and_removes_invalid(chunk_factory) -> None:
    sources = build_source_map(
        [
            chunk_factory("a", "alpha", source="a.md"),
            chunk_factory("b", "beta", source="b.pdf", page=4),
        ]
    )

    result = validate_citations("Answer uses [S1] and [S9] twice [S1].", sources)

    assert result.clean_answer == "Answer uses [S1] and  twice [S1]."
    assert result.cited_ids == ("S1",)
    assert result.invalid_ids == ("S9",)


def test_validate_citations_normalizes_lowercase_ids(chunk_factory) -> None:
    sources = build_source_map([chunk_factory("a", "alpha")])

    result = validate_citations("Evidence [s1].", sources)

    assert result.clean_answer == "Evidence [S1]."
    assert result.cited_ids == ("S1",)


def test_validate_citations_reports_when_none_are_present(chunk_factory) -> None:
    sources = build_source_map([chunk_factory("a", "alpha")])

    result = validate_citations("No citation here.", sources)

    assert result.clean_answer == "No citation here."
    assert result.cited_ids == ()
    assert result.invalid_ids == ()


def test_format_sources_marks_cited_and_retrieved(chunk_factory) -> None:
    sources = build_source_map(
        [
            chunk_factory("a", "alpha", source="a.md"),
            chunk_factory("b", "beta", source="b.pdf", page=4),
        ]
    )

    rendered = format_sources(sources, cited_ids={"S2"})

    assert "[S1] a.md" in rendered
    assert "[S2] b.pdf, p.4" in rendered
    assert "cited" in rendered
    assert "retrieved" in rendered
