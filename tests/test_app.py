from __future__ import annotations

from argparse import Namespace

import pytest

from app import _apply_cli_settings
from config import ConfigError


def _args(*, local_only=False, allow_cloud_rag=False) -> Namespace:
    return Namespace(local_only=local_only, allow_cloud_rag=allow_cloud_rag)


def test_cli_local_only_overrides_cloud_configuration(tmp_path, make_settings) -> None:
    settings = make_settings(
        tmp_path,
        local_only=False,
        allow_cloud_fallback=True,
        allow_cloud_rag=True,
    )

    result = _apply_cli_settings(settings, _args(local_only=True))

    assert result.local_only is True
    assert result.allow_cloud_fallback is False
    assert result.allow_cloud_rag is False


def test_cli_cloud_rag_is_explicit(tmp_path, make_settings) -> None:
    settings = make_settings(tmp_path, local_only=True)

    result = _apply_cli_settings(settings, _args(allow_cloud_rag=True))

    assert result.local_only is False
    assert result.allow_cloud_fallback is True
    assert result.allow_cloud_rag is True


def test_cli_conflicting_privacy_flags_are_rejected(tmp_path, make_settings) -> None:
    settings = make_settings(tmp_path)

    with pytest.raises(ConfigError):
        _apply_cli_settings(
            settings, _args(local_only=True, allow_cloud_rag=True)
        )
