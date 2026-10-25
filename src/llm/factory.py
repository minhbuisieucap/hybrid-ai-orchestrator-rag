"""Construct only the providers that are actually configured."""

from __future__ import annotations

from config import Settings
from src.llm.base import LLMProvider
from src.llm.gemini_provider import GeminiProvider
from src.llm.ollama_provider import OllamaProvider
from src.llm.openai_provider import OpenAIProvider


def build_providers(settings: Settings) -> dict[str, LLMProvider]:
    providers: dict[str, LLMProvider] = {}
    if settings.ollama_model:
        providers["ollama"] = OllamaProvider(
            host=settings.ollama_host,
            model=settings.ollama_model,
            timeout=settings.request_timeout_seconds,
        )
    if settings.openai_api_key and settings.openai_model:
        providers["openai"] = OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            timeout=settings.request_timeout_seconds,
        )
    if settings.gemini_api_key and settings.gemini_model:
        providers["gemini"] = GeminiProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            timeout=settings.request_timeout_seconds,
        )
    return providers
