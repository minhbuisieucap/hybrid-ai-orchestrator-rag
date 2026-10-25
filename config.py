"""Central configuration for the hybrid RAG chatbot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent


class ConfigError(ValueError):
    """Raised when configuration values are invalid or contradictory."""


def is_loopback_url(value: str) -> bool:
    """Return True only for endpoints that stay on this machine."""
    parsed = urlparse(value.strip())
    return parsed.hostname in {"localhost", "127.0.0.1", "::1"}


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip().lower() for part in value.split(",") if part.strip())


def _bool(value: str | bool | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"Expected a boolean, got {value!r}")


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    storage_dir: Path
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    chunk_size: int = 800
    chunk_overlap: int = 120
    dense_top_k: int = 8
    bm25_top_k: int = 8
    fused_top_k: int = 5
    rrf_k: int = 60
    dense_min_similarity: float = 0.30
    bm25_min_score: float = 0.0
    bm25_min_coverage: float = 0.67
    max_context_chars: int = 6000
    local_only: bool = True
    allow_cloud_fallback: bool = False
    allow_cloud_rag: bool = False
    default_provider: str = "ollama"
    provider_priority: tuple[str, ...] = ("ollama", "openai", "gemini")
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:1.7b"
    openai_model: str = ""
    gemini_model: str = ""
    openai_api_key: str = field(default="", repr=False)
    gemini_api_key: str = field(default="", repr=False)
    request_timeout_seconds: float = 60.0
    llm_temperature: float = 0.2
    history_messages: int = 6

    @property
    def chunks_file(self) -> Path:
        return self.storage_dir / "chunks.json"

    @property
    def embeddings_file(self) -> Path:
        return self.storage_dir / "embeddings.npy"

    @property
    def manifest_file(self) -> Path:
        return self.storage_dir / "manifest.json"

    @property
    def cloud_enabled_for_general(self) -> bool:
        return (not self.local_only) and self.allow_cloud_fallback

    @property
    def cloud_enabled_for_documents(self) -> bool:
        return self.cloud_enabled_for_general and self.allow_cloud_rag

    def validate(self) -> None:
        if not self.embedding_model.strip():
            raise ConfigError("EMBEDDING_MODEL must not be empty")
        if self.chunk_size <= 0:
            raise ConfigError("CHUNK_SIZE must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ConfigError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
        for name, value in (
            ("DENSE_TOP_K", self.dense_top_k),
            ("BM25_TOP_K", self.bm25_top_k),
            ("FUSED_TOP_K", self.fused_top_k),
            ("RRF_K", self.rrf_k),
            ("MAX_CONTEXT_CHARS", self.max_context_chars),
            ("EMBEDDING_BATCH_SIZE", self.embedding_batch_size),
        ):
            if value <= 0:
                raise ConfigError(f"{name} must be positive")
        if self.history_messages < 0:
            raise ConfigError("HISTORY_MESSAGES must be >= 0")
        if not -1.0 <= self.dense_min_similarity <= 1.0:
            raise ConfigError("DENSE_MIN_SIMILARITY must be between -1 and 1")
        if not 0.0 <= self.bm25_min_coverage <= 1.0:
            raise ConfigError("BM25_MIN_COVERAGE must be between 0 and 1")
        if self.request_timeout_seconds <= 0:
            raise ConfigError("REQUEST_TIMEOUT_SECONDS must be positive")
        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ConfigError("LLM_TEMPERATURE must be between 0 and 2")
        if not self.ollama_host.strip():
            raise ConfigError("OLLAMA_HOST must not be empty")
        if not urlparse(self.ollama_host).hostname:
            raise ConfigError("OLLAMA_HOST must be a valid URL")
        if self.local_only and not is_loopback_url(self.ollama_host):
            raise ConfigError(
                "LOCAL_ONLY=true requires OLLAMA_HOST to use localhost, "
                "127.0.0.1, or ::1"
            )
        allowed_providers = {"ollama", "openai", "gemini"}
        if not self.provider_priority:
            raise ConfigError("PROVIDER_PRIORITY must contain at least one provider")
        unknown = set(self.provider_priority) - allowed_providers
        if unknown:
            raise ConfigError(
                "Unknown provider(s) in PROVIDER_PRIORITY: "
                + ", ".join(sorted(unknown))
            )
        if self.default_provider.lower() not in self.provider_priority:
            raise ConfigError("DEFAULT_PROVIDER must appear in PROVIDER_PRIORITY")
        if self.local_only and "ollama" not in self.provider_priority:
            raise ConfigError(
                "LOCAL_ONLY=true requires ollama in PROVIDER_PRIORITY"
            )
        if self.local_only and not self.ollama_model:
            raise ConfigError("LOCAL_ONLY=true requires a non-empty OLLAMA_MODEL")
        if self.local_only and (self.allow_cloud_fallback or self.allow_cloud_rag):
            raise ConfigError(
                "LOCAL_ONLY=true cannot be combined with cloud fallback or cloud RAG"
            )
        if self.allow_cloud_rag and not self.allow_cloud_fallback:
            raise ConfigError(
                "ALLOW_CLOUD_RAG=true requires ALLOW_CLOUD_FALLBACK=true"
            )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        project_root: Path | None = None,
    ) -> "Settings":
        root = (project_root or PROJECT_ROOT).resolve()
        if env is None:
            try:
                from dotenv import load_dotenv

                load_dotenv(root / ".env")
            except ImportError:
                pass
            values = os.environ
        else:
            values = env

        def path_value(name: str, default: str) -> Path:
            raw = Path(values.get(name, default)).expanduser()
            return raw if raw.is_absolute() else root / raw

        settings = cls(
            project_root=root,
            data_dir=path_value("DOCUMENTS_DIR", "data/documents"),
            storage_dir=path_value("STORAGE_DIR", "storage"),
            embedding_model=values.get(
                "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            ).strip(),
            embedding_batch_size=int(values.get("EMBEDDING_BATCH_SIZE", "32")),
            chunk_size=int(values.get("CHUNK_SIZE", "800")),
            chunk_overlap=int(values.get("CHUNK_OVERLAP", "120")),
            dense_top_k=int(values.get("DENSE_TOP_K", "8")),
            bm25_top_k=int(values.get("BM25_TOP_K", "8")),
            fused_top_k=int(values.get("FUSED_TOP_K", "5")),
            rrf_k=int(values.get("RRF_K", "60")),
            dense_min_similarity=float(
                values.get("DENSE_MIN_SIMILARITY", "0.30")
            ),
            bm25_min_score=float(values.get("BM25_MIN_SCORE", "0.0")),
            bm25_min_coverage=float(values.get("BM25_MIN_COVERAGE", "0.67")),
            max_context_chars=int(values.get("MAX_CONTEXT_CHARS", "6000")),
            local_only=_bool(values.get("LOCAL_ONLY"), True),
            allow_cloud_fallback=_bool(
                values.get("ALLOW_CLOUD_FALLBACK"), False
            ),
            allow_cloud_rag=_bool(values.get("ALLOW_CLOUD_RAG"), False),
            default_provider=values.get("DEFAULT_PROVIDER", "ollama").strip().lower(),
            provider_priority=_split_csv(
                values.get(
                    "PROVIDER_PRIORITY", "ollama,openai,gemini"
                )
            ),
            ollama_host=values.get(
                "OLLAMA_HOST", "http://localhost:11434"
            ).strip(),
            ollama_model=values.get("OLLAMA_MODEL", "qwen3:1.7b").strip(),
            openai_model=values.get("OPENAI_MODEL", "").strip(),
            gemini_model=values.get("GEMINI_MODEL", "").strip(),
            openai_api_key=values.get("OPENAI_API_KEY", "").strip(),
            gemini_api_key=values.get("GEMINI_API_KEY", "").strip(),
            request_timeout_seconds=float(
                values.get("REQUEST_TIMEOUT_SECONDS", "60")
            ),
            llm_temperature=float(values.get("LLM_TEMPERATURE", "0.2")),
            history_messages=int(values.get("HISTORY_MESSAGES", "6")),
        )
        settings.validate()
        return settings

    def with_overrides(self, **overrides) -> "Settings":
        settings = replace(self, **overrides)
        settings.validate()
        return settings
