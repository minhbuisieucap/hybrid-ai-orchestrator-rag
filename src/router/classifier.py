"""Deterministic query-intent classification.

Intent classification is intentionally separate from provider selection. A
document cue is enough to trigger retrieval even for a very short question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Route(str, Enum):
    GENERAL = "general"
    DOCUMENT = "document"
    DOCUMENT_REASONING = "document_reasoning"


@dataclass(frozen=True)
class Classification:
    route: Route
    reason: str
    explicit_document: bool


_DOCUMENT_PATTERNS = (
    r"\bmy (?:pdf|document|documents|files?|notes)\b",
    r"\b(?:the|this|these|those|uploaded|local) (?:pdf|document|documents|files?|notes)\b",
    r"\b(?:knowledge base|local corpus|local index)\b",
    r"\baccording to (?:the|my|these|those)\b",
    r"\b(?:document|documents|pdf|file|files) (?:say|says|state|states|mention|mentions)\b",
    r"\b(?:policy|policies|notes?) (?:say|says|state|states)\b",
    r"\bin the (?:pdf|document|documents|files?|notes)\b",
    r"\bfrom the (?:pdf|document|documents|files?|notes)\b",
    r"\.(?:pdf|txt|md)\b",
)

_REASONING_PATTERN = re.compile(
    r"\b(?:compare|contrast|analy[sz]e|evaluate|explain|why|reason|"
    r"trade-?off|implications?|contradict|derive|critique)\b",
    re.IGNORECASE,
)


def _has_document_cue(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in _DOCUMENT_PATTERNS)


def classify_question(question: str) -> Classification:
    text = " ".join(question.strip().split())
    if not text:
        return Classification(
            route=Route.GENERAL,
            reason="empty question defaults to general conversation",
            explicit_document=False,
        )

    if not _has_document_cue(text.lower()):
        return Classification(
            route=Route.GENERAL,
            reason="no explicit document reference",
            explicit_document=False,
        )

    if _REASONING_PATTERN.search(text):
        return Classification(
            route=Route.DOCUMENT_REASONING,
            reason="explicit document reference plus reasoning keyword",
            explicit_document=True,
        )
    return Classification(
        route=Route.DOCUMENT,
        reason="explicit document reference",
        explicit_document=True,
    )
