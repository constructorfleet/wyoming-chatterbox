"""Tests for configuration settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wyoming_chatterbox.config import Settings


def test_defaults():
    s = Settings()
    assert s.wyoming_host == "0.0.0.0"
    assert s.wyoming_port == 10200
    assert s.chatterbox_variant == "multilingual"
    assert s.chatterbox_device == "auto"
    assert s.chatterbox_preload is True
    assert s.chatterbox_streaming_mode == "segmented"
    assert s.chatterbox_seed is None
    assert s.log_level == "INFO"


def test_env_var_parsing(monkeypatch):
    monkeypatch.setenv("WYOMING_PORT", "12345")
    monkeypatch.setenv("CHATTERBOX_VARIANT", "turbo")
    monkeypatch.setenv("CHATTERBOX_DEVICE", "cpu")
    monkeypatch.setenv("CHATTERBOX_SEED", "42")
    s = Settings()
    assert s.wyoming_port == 12345
    assert s.chatterbox_variant == "turbo"
    assert s.chatterbox_device == "cpu"
    assert s.chatterbox_seed == 42


def test_case_insensitive_env(monkeypatch):
    monkeypatch.setenv("chatterbox_device", "mps")
    s = Settings()
    assert s.chatterbox_device == "mps"


def test_active_variants_single():
    s = Settings(chatterbox_variant="nano")
    assert s.active_variants == ["nano"]


def test_active_variants_multi():
    s = Settings(chatterbox_variants="standard, turbo ,nano")
    assert s.active_variants == ["standard", "turbo", "nano"]


def test_active_variants_overrides_single():
    s = Settings(chatterbox_variant="nano", chatterbox_variants="standard,turbo")
    assert s.active_variants == ["standard", "turbo"]


def test_device_validation():
    with pytest.raises(ValidationError):
        Settings(chatterbox_device="gpu")


def test_variant_validation():
    with pytest.raises(ValidationError):
        Settings(chatterbox_variant="giant")


def test_variants_list_validation():
    with pytest.raises(ValidationError):
        Settings(chatterbox_variants="standard,unknown")


def test_streaming_mode_validation():
    with pytest.raises(ValidationError):
        Settings(chatterbox_streaming_mode="fast")


def test_positive_validation():
    with pytest.raises(ValidationError):
        Settings(wyoming_port=0)
    with pytest.raises(ValidationError):
        Settings(chatterbox_synthesis_workers=-1)


def test_log_format_validation():
    with pytest.raises(ValidationError):
        Settings(log_format="xml")


def test_min_max_chars_validation():
    with pytest.raises(ValidationError):
        Settings(chatterbox_segment_min_chars=500, chatterbox_segment_max_chars=100)
