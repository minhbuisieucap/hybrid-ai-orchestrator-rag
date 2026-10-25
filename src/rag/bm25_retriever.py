"""Lexical BM25 retrieval with a shared Unicode tokenizer."""

from __future__ import annotations

import re
from typing import Sequence

from rank_bm25 import BM25Plus

from src.rag.types import Chunk, RankedChunk

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "according",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "document",
        "documents",
        "file",
        "files",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "knowledge",
        "local",
        "my",
        "of",
        "on",
        "or",
        "pdf",
        "pdfs",
        "said",
        "sample",
        "say",
        "says",
        "that",
        "the",
        "this",
        "to",
        "uploaded",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)


def _normalize_token(token: str) -> str:
    """Very small plural normalization, not a full stemmer."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if (
        len(token) > 3
        and token.endswith("s")
        and not token.endswith(("ss", "us", "is"))
    ):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    tokens = []
    for match in _TOKEN_RE.finditer(text):
        token = _normalize_token(match.group(0).casefold())
        if token not in _STOPWORDS:
            tokens.append(token)
    return tokens


def token_coverage(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    document_tokens = set(tokenize(text))
    return len(query_tokens & document_tokens) / len(query_tokens)


class BM25Retriever:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self._corpus = [tokenize(chunk.text) for chunk in self.chunks]
        has_lexical_content = any(self._corpus)
        self._bm25 = BM25Plus(self._corpus) if has_lexical_content else None

    def search(self, query: str, *, top_k: int) -> list[RankedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_tokens = tokenize(query)
        if not query_tokens or self._bm25 is None:
            return []

        query_terms = set(query_tokens)
        scores = [float(score) for score in self._bm25.get_scores(query_tokens)]
        order = sorted(
            (
                index
                for index, score in enumerate(scores)
                if score > 0
                and query_terms & set(self._corpus[index])
            ),
            key=lambda index: (-scores[index], self.chunks[index].chunk_id),
        )[: min(top_k, len(self.chunks))]

        return [
            RankedChunk(
                chunk=self.chunks[index],
                score=scores[index],
                rank=position + 1,
            )
            for position, index in enumerate(order)
        ]

    def query_coverage(self, query: str, chunk: Chunk) -> float:
        return token_coverage(query, chunk.text)
