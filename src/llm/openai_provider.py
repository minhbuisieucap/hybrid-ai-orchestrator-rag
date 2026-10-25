"""OpenAI Responses API provider."""

from __future__ import annotations

from src.llm.base import (
    GenerationRequest,
    GenerationResult,
    LLMProvider,
    ProviderError,
)


class OpenAIProvider(LLMProvider):
    name = "openai"
    is_local = False

    def __init__(self, *, api_key: str, model: str, timeout: float) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ProviderError(
                    "The openai package is not installed. Install requirements first."
                ) from exc
            self._client = OpenAI(
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=0,
            )
        return self._client

    def generate(self, request: GenerationRequest) -> GenerationResult:
        payload = {
            "model": self.model,
            "input": request.prompt,
            "store": False,
            "timeout": self.timeout,
        }
        if request.system_prompt:
            payload["instructions"] = request.system_prompt
        if request.max_tokens is not None:
            payload["max_output_tokens"] = request.max_tokens
        try:
            response = self._get_client().responses.create(**payload)
            text = response.output_text
        except Exception as exc:
            message = " ".join(str(exc).split())[:200]
            raise ProviderError(
                f"OpenAI request failed: {message or exc.__class__.__name__}"
            ) from exc

        if not text or not text.strip():
            raise ProviderError("OpenAI returned an empty response")
        return GenerationResult(
            text=text.strip(),
            provider=self.name,
            model=self.model,
            is_local=False,
        )
