from __future__ import annotations

import pytest

from src.rag.chunker import chunk_text


def test_blank_text_produces_no_chunks() -> None:
    assert chunk_text("   \n\t", source="empty.md", chunk_size=20, overlap=5) == []


def test_short_text_produces_one_chunk_with_metadata() -> None:
    chunks = chunk_text(
        "A short policy paragraph.",
        source="policies/refund.md",
        page=None,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) == 1
    assert chunks[0].source == "policies/refund.md"
    assert chunks[0].page is None
    assert chunks[0].chunk_id == "policies/refund.md#p0#c0"
    assert chunks[0].chunk_index == 0
    assert chunks[0].char_start == 0
    assert chunks[0].text == "A short policy paragraph."


def test_overlapping_chunks_have_expected_starts() -> None:
    chunks = chunk_text(
        "abcdefghijklmnop",
        source="a.txt",
        chunk_size=5,
        overlap=2,
    )

    assert [c.text for c in chunks] == ["abcde", "defgh", "ghijk", "jklmn", "mnop"]
    assert [c.char_start for c in chunks] == [0, 3, 6, 9, 12]
    assert all(c.char_end - c.char_start <= 5 for c in chunks)


def test_pdf_page_number_and_ids_are_preserved() -> None:
    chunks = chunk_text(
        "Page text for citations.",
        source="manual.pdf",
        page=3,
        chunk_size=100,
        overlap=20,
    )

    assert chunks[0].page == 3
    assert chunks[0].chunk_id == "manual.pdf#p3#c0"


def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError):
        chunk_text("text", source="a.txt", chunk_size=10, overlap=10)
