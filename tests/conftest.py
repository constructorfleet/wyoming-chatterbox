"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from wyoming_chatterbox.config import Settings
from wyoming_chatterbox.models.base import ChatterboxBackend


class FakeBackend(ChatterboxBackend):
    """A fully in-memory backend that never imports chatterbox."""

    variant = "standard"
    sample_rate = 24000

    def __init__(self, languages: list[str] | None = None, samples: int = 2400) -> None:
        self._loaded = False
        self._languages = languages or ["en"]
        self._samples = samples
        self.load_count = 0
        self.generate_calls: list[tuple[str, dict]] = []

    def load(self) -> None:
        self.load_count += 1
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def generate(self, text: str, **kwargs) -> np.ndarray:
        self.generate_calls.append((text, kwargs))
        if not self._loaded:
            self.load()
        return np.zeros(self._samples, dtype=np.float32)

    def supports_language(self, lang: str) -> bool:
        return lang.lower() in self._languages

    def supported_languages(self) -> list[str]:
        return list(self._languages)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        chatterbox_variant="standard",
        chatterbox_device="cpu",
        chatterbox_preload=False,
        chatterbox_voices_dir=str(tmp_path / "voices"),
        chatterbox_streaming_mode="buffered",
        wyoming_port=10200,
    )


@pytest.fixture
def fake_backend() -> FakeBackend:
    backend = FakeBackend()
    backend.load()
    return backend
