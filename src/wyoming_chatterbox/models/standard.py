"""Standard (English-only) Chatterbox TTS backend."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from wyoming_chatterbox.config import Settings
from wyoming_chatterbox.metrics import count_voice_preparation_cache, observe_voice_preparation
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
        self._prepared_audio_prompt_key: tuple[str, float] | None = None

    # -- lifecycle --------------------------------------------------------

    def load(self) -> None:
        from chatterbox.tts import ChatterboxTTS  # lazy import — not available at test time

        self._model = ChatterboxTTS.from_pretrained(device=self._device)

    def unload(self) -> None:
        self._model = None
        self._prepared_audio_prompt_key = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # -- generation -------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            self.load()

    def warmup_voice(self, voice_path: str) -> None:
        self._prepare_audio_prompt(voice_path, phase="warmup")

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
        return {
            "exaggeration": self._settings.chatterbox_exaggeration,
            "cfg_weight": min(self._settings.chatterbox_cfg_weight, _MAX_CFG_WEIGHT),
            "temperature": self._settings.chatterbox_temperature,
        }

    def _generate_with_model_prompt_cache(self, text: str, gen_kwargs: dict[str, object]) -> Any:
        assert self._model is not None  # satisfied by _ensure_loaded
        audio_prompt_path = gen_kwargs.get("audio_prompt_path")
        prepare_conditionals: Any = getattr(self._model, "prepare_conditionals", None)
        if not audio_prompt_path or prepare_conditionals is None:
            return self._model.generate(text, **gen_kwargs)  # type: ignore[union-attr]

        expected_key = (
            str(Path(str(audio_prompt_path))),
            float(self._settings.chatterbox_exaggeration),
        )

        def _prepare_conditionals_with_cache(
            prompt_path: str, *args: object, **kwargs: object
        ) -> None:
            exaggeration = kwargs.get("exaggeration")
            current_exaggeration = self._settings.chatterbox_exaggeration
            if isinstance(exaggeration, (int, float, str)):
                current_exaggeration = float(exaggeration)
            current_key = (str(Path(prompt_path)), float(current_exaggeration))
            if current_key == expected_key:
                return None
            prepare_conditionals(prompt_path, *args, **kwargs)
            return None

        self._model.prepare_conditionals = _prepare_conditionals_with_cache  # type: ignore[union-attr]
        try:
            return self._model.generate(text, **gen_kwargs)  # type: ignore[union-attr]
        finally:
            self._model.prepare_conditionals = prepare_conditionals  # type: ignore[union-attr]

    def generate(self, text: str, **kwargs: object) -> np.ndarray:
        self._ensure_loaded()
        gen_kwargs = self._build_generate_kwargs()
        gen_kwargs.update(kwargs)
        audio_prompt_path = gen_kwargs.get("audio_prompt_path")
        if audio_prompt_path:
            self._prepare_audio_prompt(str(audio_prompt_path), phase="request")
        gen_kwargs.pop("language", None)  # not supported by the standard model
        assert self._model is not None  # satisfied by _ensure_loaded
        audio: Any = self._generate_with_model_prompt_cache(text, gen_kwargs)
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        return np.asarray(audio, dtype=np.float32)

    # -- language ---------------------------------------------------------

    def supported_languages(self) -> list[str]:
        return ["en"]

    def supports_language(self, lang: str) -> bool:
        return lang.lower() == "en"
