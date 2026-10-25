from __future__ import annotations

from pathlib import Path

from src.rag.ingestion import ingest_directory


def test_ingests_supported_files_recursively_and_ignores_unsupported(tmp_path: Path) -> None:
    data = tmp_path / "documents"
    (data / "policies").mkdir(parents=True)
    (data / "policy.md").write_text(
        "# Refund policy\nRefunds are available within 30 days.", encoding="utf-8"
    )
    (data / "policies" / "warranty.txt").write_text(
        "The warranty lasts one year.", encoding="utf-8"
    )
    (data / "notes.csv").write_text("not supported", encoding="utf-8")

    result = ingest_directory(data, chunk_size=100, overlap=20)

    sources = {chunk.source for chunk in result.chunks}
    assert sources == {"policy.md", "policies/warranty.txt"}
    assert result.issues == []


def test_empty_file_is_reported_without_stopping_other_documents(tmp_path: Path) -> None:
    data = tmp_path / "documents"
    data.mkdir()
    (data / "empty.txt").write_text("   ", encoding="utf-8")
    (data / "valid.txt").write_text("Useful text.", encoding="utf-8")

    result = ingest_directory(data, chunk_size=100, overlap=20)

    assert [chunk.source for chunk in result.chunks] == ["valid.txt"]
    assert len(result.issues) == 1
    assert result.issues[0].source == "empty.txt"
    assert "no extractable text" in result.issues[0].message.lower()


def test_malformed_pdf_is_reported_without_stopping_other_documents(
    tmp_path: Path,
) -> None:
    data = tmp_path / "documents"
    data.mkdir()
    (data / "broken.pdf").write_bytes(b"not a real pdf")
    (data / "valid.md").write_text("Still readable.", encoding="utf-8")

    result = ingest_directory(data, chunk_size=100, overlap=20)

    assert [chunk.source for chunk in result.chunks] == ["valid.md"]
    assert len(result.issues) == 1
    assert result.issues[0].source == "broken.pdf"


def test_missing_directory_returns_empty_result(tmp_path: Path) -> None:
    result = ingest_directory(tmp_path / "missing", chunk_size=100, overlap=20)

    assert result.chunks == []
    assert result.issues == []
