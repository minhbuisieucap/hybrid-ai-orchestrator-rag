"""Resolve a query into a retrieval path, including document follow-ups."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.router.classifier import Classification, Route, classify_question

_FOLLOW_UP_RE = re.compile(
    r"^\s*(?:what about|what else|and\b|also\b|why\b|how\b|then\b|"
    r"does that|is that|tell me more|can you clarify|explain (?:that|this|it))",
    re.IGNORECASE,
)
_PRONOUN_RE = re.compile(r"\b(?:it|that|this|they|them|those|these)\b", re.IGNORECASE)
_SHORT_WH_RE = re.compile(
    r"^\s*(?:what|who|when|where|which|why|how)\b", re.IGNORECASE
)
_REASONING_RE = re.compile(
    r"\b(?:compare|contrast|analy[sz]e|evaluate|explain|why|reason|"
    r"trade-?off|implications?|contradict)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class IntentDecision:
    route: Route
    reason: str
    retrieval_query: str
    explicit_document: bool


def decide_intent(
    question: str,
    *,
    previous_route: Route | None = None,
    previous_question: str | None = None,
) -> IntentDecision:
    classification: Classification = classify_question(question)
    if (
        classification.route is Route.GENERAL
        and previous_route in {Route.DOCUMENT, Route.DOCUMENT_REASONING}
        and previous_question
        and (
            _FOLLOW_UP_RE.search(question)
            or (
                len(question.split()) <= 12
                and (_PRONOUN_RE.search(question) or _SHORT_WH_RE.search(question))
            )
        )
    ):
        route = (
            Route.DOCUMENT_REASONING
            if _REASONING_RE.search(question)
            else Route.DOCUMENT
        )
        return IntentDecision(
            route=route,
            reason="follow-up to the previous document-grounded answer",
            retrieval_query=f"{previous_question} {question}".strip(),
            explicit_document=False,
        )

    return IntentDecision(
        route=classification.route,
        reason=classification.reason,
        retrieval_query=question.strip(),
        explicit_document=classification.explicit_document,
    )
