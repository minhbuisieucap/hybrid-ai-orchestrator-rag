from __future__ import annotations

import pytest

from src.llm.base import (
    AllProvidersFailed,
    GenerationRequest,
    GenerationResult,
    LLMProvider,
    ProviderError,
)
from src.router.router import ProviderRouter


class FakeProvider(LLMProvider):
    def __init__(
        self,
        name: str,
        is_local: bool,
        *,
        fail: bool = False,
        text: str | None = None,
    ) -> None:
        self.name = name
        self.model = f"{name}-model"
        self.is_local = is_local
        self.fail = fail
        self.text = text or f"answer from {name}"
        self.calls: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        if self.fail:
            raise ProviderError(f"{self.name} unavailable")
        return GenerationResult(
            text=self.text,
            provider=self.name,
            model=self.model,
            is_local=self.is_local,
        )


def _router(providers, **overrides):
    values = dict(
        priority=("local", "cloud"),
        default_provider="local",
        local_only=True,
        allow_cloud_fallback=False,
        allow_cloud_rag=False,
    )
    values.update(overrides)
    return ProviderRouter(providers=providers, **values)


def test_local_only_never_includes_cloud() -> None:
    local = FakeProvider("local", True)
    cloud = FakeProvider("cloud", False)

    router = _router(
        {"local": local, "cloud": cloud},
        local_only=True,
        allow_cloud_fallback=True,
        allow_cloud_rag=True,
    )

    assert [p.name for p in router.chain_for(document_task=False)] == ["local"]
    assert [p.name for p in router.chain_for(document_task=True)] == ["local"]


def test_cloud_requires_explicit_general_permission() -> None:
    local = FakeProvider("local", True)
    cloud = FakeProvider("cloud", False)

    router = _router(
        {"local": local, "cloud": cloud},
        local_only=False,
        allow_cloud_fallback=False,
    )

    assert [p.name for p in router.chain_for(document_task=False)] == ["local"]


def test_document_task_requires_cloud_rag_permission() -> None:
    local = FakeProvider("local", True)
    cloud = FakeProvider("cloud", False)

    router = _router(
        {"local": local, "cloud": cloud},
        local_only=False,
        allow_cloud_fallback=True,
        allow_cloud_rag=False,
    )

    assert [p.name for p in router.chain_for(document_task=False)] == [
        "local",
        "cloud",
    ]
    assert [p.name for p in router.chain_for(document_task=True)] == ["local"]


def test_cloud_rag_permission_enables_cloud_and_calls_disclosure_hook() -> None:
    local = FakeProvider("local", True)
    cloud = FakeProvider("cloud", False)
    disclosed: list[str] = []

    router = _router(
        {"local": local, "cloud": cloud},
        local_only=False,
        allow_cloud_fallback=True,
        allow_cloud_rag=True,
        before_cloud_call=lambda provider, document_task: disclosed.append(provider.name),
    )

    chain = router.chain_for(document_task=True)
    assert [p.name for p in chain] == ["local", "cloud"]

    response = router.generate(
        GenerationRequest(prompt="private question"), document_task=True
    )

    assert response.result.provider == "local"
    assert disclosed == []


def test_fallback_records_failure_and_uses_next_provider() -> None:
    local = FakeProvider("local", True, fail=True)
    cloud = FakeProvider("cloud", False)

    router = _router(
        {"local": local, "cloud": cloud},
        local_only=False,
        allow_cloud_fallback=True,
        allow_cloud_rag=True,
    )

    response = router.generate(GenerationRequest(prompt="question"))

    assert response.result.provider == "cloud"
    assert response.fallback_occurred is True
    assert response.failures[0].provider == "local"
    assert "unavailable" in response.failures[0].error


def test_cloud_disclosure_hook_runs_before_document_cloud_call() -> None:
    local = FakeProvider("local", True, fail=True)
    cloud = FakeProvider("cloud", False)
    events: list[str] = []

    def before(provider, document_task):
        events.append(f"disclosed:{provider.name}:{document_task}")

    router = _router(
        {"local": local, "cloud": cloud},
        local_only=False,
        allow_cloud_fallback=True,
        allow_cloud_rag=True,
        before_cloud_call=before,
    )

    response = router.generate(
        GenerationRequest(prompt="private question"), document_task=True
    )

    assert response.result.provider == "cloud"
    assert events == ["disclosed:cloud:True"]


def test_general_cloud_disclosure_receives_non_document_context() -> None:
    local = FakeProvider("local", True, fail=True)
    cloud = FakeProvider("cloud", False)
    events: list[tuple[str, bool]] = []

    router = _router(
        {"local": local, "cloud": cloud},
        local_only=False,
        allow_cloud_fallback=True,
        allow_cloud_rag=True,
        before_cloud_call=lambda provider, document_task: events.append(
            (provider.name, document_task)
        ),
    )

    router.generate(GenerationRequest(prompt="general question"))

    assert events == [("cloud", False)]


def test_all_providers_failed_raises_clear_error() -> None:
    local = FakeProvider("local", True, fail=True)
    router = _router({"local": local})

    with pytest.raises(AllProvidersFailed) as exc:
        router.generate(GenerationRequest(prompt="question"))

    assert "local" in str(exc.value)
