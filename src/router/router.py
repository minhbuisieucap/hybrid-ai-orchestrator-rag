"""Provider selection, explicit cloud policy, and fallback execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from src.llm.base import (
    AllProvidersFailed,
    GenerationRequest,
    GenerationResult,
    LLMProvider,
)


@dataclass(frozen=True)
class ProviderFailure:
    provider: str
    error: str


@dataclass(frozen=True)
class ProviderResponse:
    result: GenerationResult
    fallback_occurred: bool
    failures: tuple[ProviderFailure, ...]


class ProviderRouter:
    def __init__(
        self,
        *,
        providers: Mapping[str, LLMProvider],
        priority: Sequence[str],
        default_provider: str,
        local_only: bool,
        allow_cloud_fallback: bool,
        allow_cloud_rag: bool,
        before_cloud_call: Callable[[LLMProvider, bool], None] | None = None,
    ) -> None:
        self.providers = dict(providers)
        self.priority = tuple(priority)
        self.default_provider = default_provider
        self.local_only = local_only
        self.allow_cloud_fallback = allow_cloud_fallback
        self.allow_cloud_rag = allow_cloud_rag
        self.before_cloud_call = before_cloud_call

    def _ordered_names(self) -> list[str]:
        candidates = [self.default_provider]
        candidates.extend(name for name in self.priority if name != self.default_provider)
        return [name for name in candidates if name in self.providers]

    def chain_for(self, *, document_task: bool) -> list[LLMProvider]:
        allow_cloud = (
            not self.local_only
            and self.allow_cloud_fallback
            and (not document_task or self.allow_cloud_rag)
        )
        chain: list[LLMProvider] = []
        for name in self._ordered_names():
            provider = self.providers[name]
            if provider.is_local or allow_cloud:
                chain.append(provider)
        return chain

    def generate(
        self,
        request: GenerationRequest,
        *,
        document_task: bool = False,
    ) -> ProviderResponse:
        chain = self.chain_for(document_task=document_task)
        if not chain:
            raise AllProvidersFailed(
                "No LLM providers are available under the current privacy settings."
            )

        failures: list[ProviderFailure] = []
        for index, provider in enumerate(chain):
            if not provider.is_local:
                if self.before_cloud_call is not None:
                    self.before_cloud_call(provider, document_task)
            try:
                result = provider.generate(request)
            except Exception as exc:
                message = " ".join(str(exc).split()) or exc.__class__.__name__
                failures.append(
                    ProviderFailure(provider=provider.name, error=message[:200])
                )
                continue
            return ProviderResponse(
                result=result,
                fallback_occurred=index > 0,
                failures=tuple(failures),
            )

        details = "; ".join(
            f"{failure.provider}: {failure.error}" for failure in failures
        )
        raise AllProvidersFailed(f"All selected providers failed. {details}")
