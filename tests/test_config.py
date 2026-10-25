from __future__ import annotations

import pytest

from config import ConfigError, Settings


def test_invalid_chunk_overlap_is_rejected(tmp_path) -> None:
    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        storage_dir=tmp_path / "storage",
        chunk_size=100,
        chunk_overlap=100,
    )

    with pytest.raises(ConfigError):
        settings.validate()


def test_cloud_rag_and_local_only_are_incompatible(tmp_path) -> None:
    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        storage_dir=tmp_path / "storage",
        local_only=True,
        allow_cloud_rag=True,
    )

    with pytest.raises(ConfigError):
        settings.validate()


def test_local_only_requires_no_api_keys(tmp_path) -> None:
    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        storage_dir=tmp_path / "storage",
        local_only=True,
        openai_api_key="",
        gemini_api_key="",
    )

    settings.validate()
    assert settings.local_only is True


def test_invalid_top_k_is_rejected(tmp_path) -> None:
    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        storage_dir=tmp_path / "storage",
        dense_top_k=0,
    )

    with pytest.raises(ConfigError):
        settings.validate()


def test_local_only_requires_loopback_ollama_host(tmp_path) -> None:
    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        storage_dir=tmp_path / "storage",
        local_only=True,
        ollama_host="http://192.168.1.25:11434",
    )

    with pytest.raises(ConfigError):
        settings.validate()


def test_local_only_accepts_loopback_ollama_host(tmp_path) -> None:
    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        storage_dir=tmp_path / "storage",
        local_only=True,
        ollama_host="http://127.0.0.1:11434",
    )

    settings.validate()


def test_settings_repr_does_not_expose_api_keys(tmp_path) -> None:
    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        storage_dir=tmp_path / "storage",
        openai_api_key="test-openai-secret",
        gemini_api_key="test-gemini-secret",
    )

    rendered = repr(settings)

    assert "test-openai-secret" not in rendered
    assert "test-gemini-secret" not in rendered
