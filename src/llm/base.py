"""Common LLM provider interface and error types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class ProviderError(RuntimeError):
    """A provider could not produce an answer."""


class ProviderUnavailable(ProviderError):
    """A provider is not configured or not reachable."""


class AllProvidersFailed(ProviderError):
    """Every provider in the selected chain failed."""


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    system_prompt: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    provider: str
    model: str
    is_local: bool


class LLMProvider(ABC):
    name: str
    model: str
    is_local: bool

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError
