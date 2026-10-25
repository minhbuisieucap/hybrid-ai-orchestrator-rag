from __future__ import annotations

import pytest

from src.router.classifier import Route, classify_question


@pytest.mark.parametrize(
    "question",
    [
        "What is the refund policy in my PDF?",
        "Summarize the warranty terms in the uploaded documents.",
        "What does the document say about retention?",
        "According to the local knowledge base, who approves leave?",
    ],
)
def test_explicit_document_questions_route_to_rag(question: str) -> None:
    result = classify_question(question)

    assert result.route in {Route.DOCUMENT, Route.DOCUMENT_REASONING}
    assert result.reason


@pytest.mark.parametrize(
    "question",
    [
        "Write a Python function that reverses a list.",
        "Why is the sky blue?",
        "Tell me a short joke.",
    ],
)
def test_general_questions_do_not_require_documents(question: str) -> None:
    result = classify_question(question)

    assert result.route is Route.GENERAL


def test_document_question_with_reasoning_is_marked_as_reasoning() -> None:
    result = classify_question(
        "Compare the refund policy in my PDF with the warranty and explain the exceptions."
    )

    assert result.route is Route.DOCUMENT_REASONING


def test_document_reference_without_reasoning_stays_document() -> None:
    result = classify_question("What is the refund policy in my PDF?")

    assert result.route is Route.DOCUMENT


def test_question_marks_alone_do_not_make_a_document_question() -> None:
    result = classify_question("What is the capital of France? Is it Paris?")

    assert result.route is Route.GENERAL
