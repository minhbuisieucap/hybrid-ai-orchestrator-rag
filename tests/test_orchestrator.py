from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.llm.base import (
    AllProvidersFailed,
    GenerationRequest,
    GenerationResult,
    LLMProvider,
    ProviderError,
)
from src.orchestration.orchestrator import Orchestrator
from src.rag.pipeline import HybridRetrievalResult
from src.rag.types import FusedHit
from src.router.classifier import Route
from src.router.router import ProviderRouter


class FakeProvider(LLMProvider):
    def __init__(
        self,
        name: str,
        is_local: bool,
        *,
        text: str = "answer",
        fail: bool = False,
    ) -> None:
        self.name = name
        self.model = f"{name}-model"
        self.is_local = is_local
        self.text = text
        self.fail = fail
        self.calls: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        if self.fail:
            raise ProviderError("not available")
        return GenerationResult(
            text=self.text,
            provider=self.name,
            model=self.model,
            is_local=self.is_local,
        )


@dataclass
class FakeRAG:
    result: HybridRetrievalResult
    queries: list[str] = field(default_factory=list)

    def retrieve(self, query: str) -> HybridRetrievalResult:
        self.queries.append(query)
        return self.result


def _providers(local: FakeProvider, cloud: FakeProvider | None = None, **kwargs):
    mapping = {"local": local}
    if cloud is not None:
        mapping["cloud"] = cloud
    values = dict(
        priority=("local", "cloud"),
        default_provider="local",
        local_only=True,
        allow_cloud_fallback=False,
        allow_cloud_rag=False,
    )
    values.update(kwargs)
    return ProviderRouter(providers=mapping, **values)


def _hit(chunk_factory) -> FusedHit:
    chunk = chunk_factory("c1", "The refund period is 30 days.", source="refund.md")
    return FusedHit(
        chunk=chunk,
        fused_score=1 / 61,
        dense_rank=1,
        bm25_rank=1,
        dense_score=0.9,
        bm25_score=3.0,
    )


def _rag_result(hit: FusedHit, sufficient: bool = True) -> HybridRetrievalResult:
    return HybridRetrievalResult(
        selected_hits=(hit,) if sufficient else (),
        fused_hits=(hit,),
        dense_hits=(),
        bm25_hits=(),
        sufficient=sufficient,
        reason="test",
        best_dense_score=0.9,
        best_bm25_score=3.0,
        best_bm25_coverage=1.0,
    )


def test_general_question_uses_provider_without_rag(
    tmp_path, make_settings, chunk_factory
) -> None:
    local = FakeProvider("local", True, text="Paris")
    settings = make_settings(tmp_path)
    orchestrator = Orchestrator(settings, _providers(local), rag=None)

    result = orchestrator.ask("What is the capital of France?")

    assert result.route is Route.GENERAL
    assert result.answer == "Paris"
    assert result.sources == ()
    assert "general" in result.answer_label.lower()
    assert len(local.calls) == 1


def test_document_question_uses_rag_and_validates_citations(
    tmp_path, make_settings, chunk_factory
) -> None:
    local = FakeProvider("local", True, text="The refund period is 30 days. [S1]")
    settings = make_settings(tmp_path)
    rag = FakeRAG(_rag_result(_hit(chunk_factory)))
    orchestrator = Orchestrator(settings, _providers(local), rag=rag)

    result = orchestrator.ask("What is the refund policy in my PDF?")

    assert result.route in {Route.DOCUMENT, Route.DOCUMENT_REASONING}
    assert result.evidence_sufficient is True
    assert result.cited_ids == ("S1",)
    assert result.invalid_citations == ()
    assert result.sources[0].chunk.source == "refund.md"
    assert "document" in result.answer_label.lower()
    assert rag.queries


def test_invalid_citation_is_removed(
    tmp_path, make_settings, chunk_factory
) -> None:
    local = FakeProvider("local", True, text="Grounded [S1] but invented [S9].")
    settings = make_settings(tmp_path)
    rag = FakeRAG(_rag_result(_hit(chunk_factory)))
    orchestrator = Orchestrator(settings, _providers(local), rag=rag)

    result = orchestrator.ask("What does the document say about refunds?")

    assert "[S9]" not in result.answer
    assert "[S1]" in result.answer
    assert result.invalid_citations == ("S9",)


def test_no_evidence_returns_clear_message_without_calling_llm(
    tmp_path, make_settings, chunk_factory
) -> None:
    local = FakeProvider("local", True)
    settings = make_settings(tmp_path)
    hit = _hit(chunk_factory)
    rag = FakeRAG(_rag_result(hit, sufficient=False))
    orchestrator = Orchestrator(settings, _providers(local), rag=rag)

    result = orchestrator.ask("What does the document say about submarines?")

    assert result.evidence_sufficient is False
    assert "do not contain enough information" in result.answer.lower()
    assert local.calls == []


def test_no_index_returns_clear_message_without_calling_llm(
    tmp_path, make_settings
) -> None:
    local = FakeProvider("local", True)
    settings = make_settings(tmp_path)
    orchestrator = Orchestrator(settings, _providers(local), rag=None)

    result = orchestrator.ask("What does my PDF say about refunds?")

    assert "no local document index" in result.answer.lower()
    assert local.calls == []


def test_local_only_never_calls_cloud_even_when_local_fails(
    tmp_path, make_settings, chunk_factory
) -> None:
    local = FakeProvider("local", True, fail=True)
    cloud = FakeProvider("cloud", False)
    settings = make_settings(tmp_path)
    rag = FakeRAG(_rag_result(_hit(chunk_factory)))
    provider_router = _providers(
        local,
        cloud,
        local_only=True,
        allow_cloud_fallback=True,
        allow_cloud_rag=True,
    )
    orchestrator = Orchestrator(settings, provider_router, rag=rag)

    with pytest.raises(AllProvidersFailed):
        orchestrator.ask("What does my PDF say about refunds?")

    assert cloud.calls == []


def test_follow_up_uses_history_in_retrieval_query(
    tmp_path, make_settings, chunk_factory
) -> None:
    local = FakeProvider("local", True, text="Answer [S1]")
    settings = make_settings(tmp_path)
    rag = FakeRAG(_rag_result(_hit(chunk_factory)))
    orchestrator = Orchestrator(settings, _providers(local), rag=rag)

    orchestrator.ask("What is the refund policy in my PDF?")
    orchestrator.ask("What about exceptions?")

    assert "What is the refund policy in my PDF?" in rag.queries[-1]


def test_general_cloud_prompt_does_not_include_document_history(
    tmp_path, make_settings, chunk_factory
) -> None:
    local = FakeProvider("local", True, text="DOCUMENT_SECRET [S1]")
    cloud = FakeProvider("cloud", False, text="A joke")
    settings = make_settings(
        tmp_path,
        local_only=False,
        allow_cloud_fallback=True,
        allow_cloud_rag=False,
    )
    rag = FakeRAG(_rag_result(_hit(chunk_factory)))
    provider_router = _providers(
        local,
        cloud,
        local_only=False,
        allow_cloud_fallback=True,
        allow_cloud_rag=False,
    )
    orchestrator = Orchestrator(settings, provider_router, rag=rag)

    orchestrator.ask("What does the document say about refunds?")
    local.fail = True
    result = orchestrator.ask("Tell me a joke.")

    assert result.answer == "A joke"
    assert cloud.calls
    assert "DOCUMENT_SECRET" not in cloud.calls[-1].prompt
    assert "What does the document say about refunds?" not in cloud.calls[-1].prompt


def test_answer_with_only_invalid_citation_is_an_error(
    tmp_path, make_settings, chunk_factory
) -> None:
    local = FakeProvider("local", True, text="[S9]")
    settings = make_settings(tmp_path)
    rag = FakeRAG(_rag_result(_hit(chunk_factory)))
    orchestrator = Orchestrator(settings, _providers(local), rag=rag)

    with pytest.raises(ProviderError):
        orchestrator.ask("What does the document say about refunds?")
