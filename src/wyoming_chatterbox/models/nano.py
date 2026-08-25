"""Nano (small/fast) Chatterbox TTS backend."""

from __future__ import annotations

import logging

from wyoming_chatterbox.config import Settings
from wyoming_chatterbox.models.standard import StandardBackend

logger = logging.getLogger(__name__)


class NanoBackend(StandardBackend):
    """Nano variant — uses the standard TTS model with a distinct variant name."""

    variant = "nano"

    def __init__(self, device: str, settings: Settings) -> None:
        super().__init__(device, settings)
