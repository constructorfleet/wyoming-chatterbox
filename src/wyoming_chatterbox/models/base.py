"""Abstract base class for Chatterbox model backends."""

from __future__ import annotations

import abc

import numpy as np


class ChatterboxBackend(abc.ABC):
    """Abstract base for all Chatterbox model variants."""

    variant: str
    sample_rate: int

    @abc.abstractmethod
    def load(self) -> None:
        """Load the model into memory."""

    @abc.abstractmethod
    def unload(self) -> None:
        """Release model from memory."""

    @property
    @abc.abstractmethod
    def is_loaded(self) -> bool:
        """Return True if the model is currently loaded."""

    @abc.abstractmethod
    def generate(self, text: str, **kwargs) -> np.ndarray:
        """Synthesize *text* and return a float32 audio array."""

    @abc.abstractmethod
    def supported_languages(self) -> list[str]:
        """Return list of BCP-47 language codes supported by this backend."""

    def supports_language(self, lang: str) -> bool:
        """Return True if *lang* (case-insensitive) is supported."""
        return lang.lower() in {lng.lower() for lng in self.supported_languages()}
