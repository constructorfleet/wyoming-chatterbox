"""Turbo (low-latency) Chatterbox TTS backend."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from wyoming_chatterbox.metrics import count_voice_preparation_cache, observe_voice_preparation
from wyoming_chatterbox.models.standard import StandardBackend

logger = logging.getLogger(__name__)

# The turbo variant is optimised for low latency; cap cfg_weight to keep
# quality acceptable at higher speeds.
_TURBO_MAX_CFG_WEIGHT: float = 0.3


class TurboBackend(StandardBackend):
    """Turbo variant — uses ChatterboxTurboTTS with capped cfg_weight."""

    variant = "turbo"

    def load(self) -> None:
        from chatterbox.tts_turbo import ChatterboxTurboTTS  # lazy import

        self._model = ChatterboxTurboTTS.from_pretrained(device=self._device)

    def _prepare_audio_prompt(self, audio_prompt_path: str, *, phase: str) -> None:
        self._ensure_loaded()
        normalized_path = str(Path(audio_prompt_path))
        key = (normalized_path, float(self._settings.chatterbox_exaggeration))
        if self._prepared_audio_prompt_key == key:
            count_voice_preparation_cache(self.variant, "hit")
            logger.debug(
                "Reused prepared voice prompt for %s (%s, phase=%s)",
                self.variant,
                normalized_path,
                phase,
            )
            return
        count_voice_preparation_cache(self.variant, "miss")
        start_time = time.perf_counter()
        assert self._model is not None  # satisfied by _ensure_loaded
        try:
            self._model.prepare_conditionals(  # type: ignore[union-attr]
                normalized_path,
                exaggeration=self._settings.chatterbox_exaggeration,
                norm_loudness=True,
            )
        except Exception:
            duration = time.perf_counter() - start_time
            observe_voice_preparation(
                variant=self.variant,
                phase=phase,
                status="error",
                duration_seconds=duration,
            )
            raise
        duration = time.perf_counter() - start_time
        observe_voice_preparation(
            variant=self.variant,
            phase=phase,
            status="success",
            duration_seconds=duration,
        )
        self._prepared_audio_prompt_key = key
        logger.info(
            "Prepared voice prompt for %s in %.1f ms (phase=%s, voice=%s)",
            self.variant,
            duration * 1000.0,
            phase,
            normalized_path,
        )

    def _build_generate_kwargs(self) -> dict[str, object]:
        kwargs = super()._build_generate_kwargs()
        kwargs["cfg_weight"] = min(float(kwargs["cfg_weight"]), _TURBO_MAX_CFG_WEIGHT)  # type: ignore[arg-type]
        kwargs["top_k"] = self._settings.chatterbox_top_k
        return kwargs
