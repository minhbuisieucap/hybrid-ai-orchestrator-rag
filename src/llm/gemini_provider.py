"""Google Gemini API provider."""

from __future__ import annotations

from src.llm.base import (
    GenerationRequest,
    GenerationResult,
    LLMProvider,
    ProviderError,
)


class GeminiProvider(LLMProvider):
    name = "gemini"
    is_local = False

    def __init__(self, *, api_key: str, model: str, timeout: float) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client = None
        self._types = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:
                raise ProviderError(
                    "The google-genai package is not installed. "
                    "Install requirements first."
                ) from exc
            self._types = types
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(
                    timeout=int(self.timeout * 1000)
                ),
            )
        return self._client

    def generate(self, request: GenerationRequest) -> GenerationResult:
        client = self._get_client()
        config = self._types.GenerateContentConfig(
            system_instruction=request.system_prompt,
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
        )
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=request.prompt,
                config=config,
            )
            text = response.text
        except Exception as exc:
            message = " ".join(str(exc).split())[:200]
            raise ProviderError(
                f"Gemini request failed: {message or exc.__class__.__name__}"
            ) from exc

        if not text or not text.strip():
            raise ProviderError("Gemini returned an empty response")
        return GenerationResult(
            text=text.strip(),
            provider=self.name,
            model=self.model,
            is_local=False,
        )
