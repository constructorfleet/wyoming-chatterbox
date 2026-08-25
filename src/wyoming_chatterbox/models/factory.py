"""Factory helpers for creating and configuring Chatterbox backends."""

from __future__ import annotations

import logging

from wyoming_chatterbox.config import Settings
from wyoming_chatterbox.models.base import ChatterboxBackend
from wyoming_chatterbox.models.multilingual import MultilingualBackend
from wyoming_chatterbox.models.nano import NanoBackend
from wyoming_chatterbox.models.standard import StandardBackend
from wyoming_chatterbox.models.turbo import TurboBackend

logger = logging.getLogger(__name__)

_BACKEND_MAP: dict[str, type[ChatterboxBackend]] = {
    "standard": StandardBackend,
    "multilingual": MultilingualBackend,
    "turbo": TurboBackend,
    "nano": NanoBackend,
}


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _mps_available() -> bool:
    try:
        import torch

        return torch.backends.mps.is_available()
    except Exception:  # noqa: BLE001
        return False


def resolve_device(device: str) -> str:
    """Resolve *device* string to a concrete device name.

    ``"auto"`` picks the best available device (cuda > mps > cpu).
    Explicit device strings raise ``RuntimeError`` if the hardware is absent.
    """
    if device == "auto":
        if _cuda_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"
    if device == "cuda" and not _cuda_available():
        raise RuntimeError("CUDA device requested but torch.cuda is not available")
    if device == "mps" and not _mps_available():
        raise RuntimeError("MPS device requested but torch.backends.mps is not available")
    return device


def create_backend(variant: str, device: str, settings: Settings) -> ChatterboxBackend:
    """Instantiate and return the backend for *variant* on *device*.

    Raises ``ValueError`` for unknown variants.
    """
    cls = _BACKEND_MAP.get(variant)
    if cls is None:
        raise ValueError(f"Unknown variant {variant!r}; must be one of {sorted(_BACKEND_MAP)}")
    return cls(device, settings)  # type: ignore[call-arg]
