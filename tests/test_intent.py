from __future__ import annotations

from src.orchestration.intent import decide_intent
from src.router.classifier import Route


def test_follow_up_after_document_answer_stays_in_document_mode() -> None:
    decision = decide_intent(
        "What about the exceptions?",
        previous_route=Route.DOCUMENT,
        previous_question="What is the refund policy?",
    )

    assert decision.route is Route.DOCUMENT
    assert "What is the refund policy?" in decision.retrieval_query
    assert decision.reason


def test_reasoning_follow_up_after_document_answer_is_document_reasoning() -> None:
    decision = decide_intent(
        "Why does that apply?",
        previous_route=Route.DOCUMENT_REASONING,
        previous_question="Compare the two policies.",
    )

    assert decision.route is Route.DOCUMENT_REASONING
    assert decision.retrieval_query


def test_no_previous_document_route_keeps_general_default() -> None:
    decision = decide_intent(
        "What about that?",
        previous_route=Route.GENERAL,
        previous_question="Tell me a joke.",
    )

    assert decision.route is Route.GENERAL


def test_explicit_document_question_overrides_general_history() -> None:
    decision = decide_intent(
        "What does my PDF say about returns?",
        previous_route=Route.GENERAL,
        previous_question="Hello",
    )

    assert decision.route in {Route.DOCUMENT, Route.DOCUMENT_REASONING}


def test_short_wh_question_after_document_answer_stays_document() -> None:
    decision = decide_intent(
        "What is the deadline?",
        previous_route=Route.DOCUMENT,
        previous_question="What does the policy say about refunds?",
    )

    assert decision.route is Route.DOCUMENT
    assert "What does the policy say about refunds?" in decision.retrieval_query
