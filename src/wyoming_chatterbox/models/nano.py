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

        try:
            self._model = ChatterboxTurboTTS.from_pretrained(device=self._device, nano=True)
        except TypeError:
            # chatterbox-tts < unreleased: from_pretrained() lacks nano param.
            # Download the nano snapshot ourselves and use from_local().
            from huggingface_hub import snapshot_download

            local_path = snapshot_download(repo_id="ResembleAI/chatterbox-nano")
            self._model = ChatterboxTurboTTS.from_local(local_path, device=self._device, nano=True)
