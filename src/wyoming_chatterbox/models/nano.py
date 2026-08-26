"""Nano (small/fast) Chatterbox TTS backend."""

from __future__ import annotations

import logging

from wyoming_chatterbox.config import Settings
from wyoming_chatterbox.models.turbo import TurboBackend

logger = logging.getLogger(__name__)


class NanoBackend(TurboBackend):
    """Nano variant — uses ChatterboxTurboTTS with nano=True."""

    variant = "nano"

    def __init__(self, device: str, settings: Settings) -> None:
        super().__init__(device, settings)

    def load(self) -> None:
        from chatterbox.tts_turbo import ChatterboxTurboTTS  # lazy import

        self._model = ChatterboxTurboTTS.from_pretrained(device=self._device, nano=True)
