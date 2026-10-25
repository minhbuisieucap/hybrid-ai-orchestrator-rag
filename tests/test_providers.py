from __future__ import annotations

from src.llm.factory import build_providers
from src.llm.ollama_provider import OllamaProvider


def test_local_only_configuration_needs_no_cloud_credentials(tmp_path, make_settings) -> None:
    settings = make_settings(
        tmp_path,
        local_only=True,
        openai_api_key="",
        gemini_api_key="",
        openai_model="",
        gemini_model="",
    )

    providers = build_providers(settings)

    assert set(providers) == {"ollama"}


def test_cloud_providers_are_only_registered_when_key_and_model_exist(
    tmp_path, make_settings
) -> None:
    settings = make_settings(
        tmp_path,
        local_only=False,
        openai_api_key="test-key",
        openai_model="test-model",
        gemini_api_key="",
        gemini_model="",
    )

    providers = build_providers(settings)

    assert set(providers) == {"ollama", "openai"}
    assert providers["openai"].is_local is False


def test_remote_ollama_endpoint_is_not_treated_as_local() -> None:
    remote = OllamaProvider(
        host="http://192.168.1.25:11434",
        model="qwen3:1.7b",
        timeout=5,
    )
    local = OllamaProvider(
        host="http://127.0.0.1:11434",
        model="qwen3:1.7b",
        timeout=5,
    )

    assert remote.is_local is False
    assert local.is_local is True
