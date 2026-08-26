"""Standard (English-only) Chatterbox TTS backend."""

from __future__ import annotations

import logging

import numpy as np

from wyoming_chatterbox.config import Settings
from wyoming_chatterbox.models.base import ChatterboxBackend

logger = logging.getLogger(__name__)

_MAX_CFG_WEIGHT: float = 1.0


class StandardBackend(ChatterboxBackend):
    """Wraps the standard English-only ``ChatterboxTTS`` model."""

    variant = "standard"
    sample_rate = 24000

    def __init__(self, device: str, settings: Settings) -> None:
        self._device = device
        self._settings = settings
        self._model: object | None = None

    # -- lifecycle --------------------------------------------------------

    def load(self) -> None:
        from chatterbox.tts import ChatterboxTTS  # lazy import — not available at test time

        self._model = ChatterboxTTS.from_pretrained(device=self._device)

    def unload(self) -> None:
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # -- generation -------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            self.load()

    def _build_generate_kwargs(self) -> dict[str, object]:
        return {
            "exaggeration": self._settings.chatterbox_exaggeration,
            "cfg_weight": min(self._settings.chatterbox_cfg_weight, _MAX_CFG_WEIGHT),
            "temperature": self._settings.chatterbox_temperature,
        }

    def generate(self, text: str, **kwargs: object) -> np.ndarray:
        self._ensure_loaded()
        gen_kwargs = self._build_generate_kwargs()
        gen_kwargs.update(kwargs)
        gen_kwargs.pop("language", None)  # not supported by the standard model
        assert self._model is not None  # satisfied by _ensure_loaded
        audio = self._model.generate(text, **gen_kwargs)  # type: ignore[union-attr]
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        return np.asarray(audio, dtype=np.float32)

    # -- language ---------------------------------------------------------

    def supported_languages(self) -> list[str]:
        return ["en"]

    def supports_language(self, lang: str) -> bool:
        return lang.lower() == "en"
