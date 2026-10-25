"""Local Ollama provider."""

from __future__ import annotations

from config import is_loopback_url
from src.llm.base import (
    GenerationRequest,
    GenerationResult,
    LLMProvider,
    ProviderError,
)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, *, host: str, model: str, timeout: float) -> None:
        self.host = host
        self.model = model
        self.timeout = timeout
        self.is_local = is_loopback_url(host)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import ollama
            except ImportError as exc:
                raise ProviderError(
                    "The ollama package is not installed. Install requirements first."
                ) from exc
            self._client = ollama.Client(host=self.host, timeout=self.timeout)
        return self._client

    def generate(self, request: GenerationRequest) -> GenerationResult:
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        options = {"temperature": request.temperature}
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens

        try:
            response = self._get_client().chat(
                model=self.model,
                messages=messages,
                options=options,
                stream=False,
            )
            text = response.message.content
        except Exception as exc:
            message = " ".join(str(exc).split())[:200]
            raise ProviderError(
                f"Ollama request failed: {message or exc.__class__.__name__}"
            ) from exc

        if not text or not text.strip():
            raise ProviderError("Ollama returned an empty response")
        return GenerationResult(
            text=text.strip(),
            provider=self.name,
            model=self.model,
            is_local=True,
        )
