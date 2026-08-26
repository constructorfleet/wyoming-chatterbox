"""Turbo (low-latency) Chatterbox TTS backend."""

from __future__ import annotations

import logging

from wyoming_chatterbox.config import Settings
from wyoming_chatterbox.models.standard import StandardBackend

logger = logging.getLogger(__name__)

# The turbo variant is optimised for low latency; cap cfg_weight to keep
# quality acceptable at higher speeds.
_TURBO_MAX_CFG_WEIGHT: float = 0.3


class TurboBackend(StandardBackend):
    """Turbo variant — uses ChatterboxTurboTTS with capped cfg_weight."""

    variant = "turbo"

    def __init__(self, device: str, settings: Settings) -> None:
        super().__init__(device, settings)

    def load(self) -> None:
        from chatterbox.tts_turbo import ChatterboxTurboTTS  # lazy import

        self._model = ChatterboxTurboTTS.from_pretrained(device=self._device)

    def _build_generate_kwargs(self) -> dict[str, object]:
        kwargs = super()._build_generate_kwargs()
        kwargs["cfg_weight"] = min(float(kwargs["cfg_weight"]), _TURBO_MAX_CFG_WEIGHT)  # type: ignore[arg-type]
        kwargs["top_k"] = self._settings.chatterbox_top_k
        return kwargs
