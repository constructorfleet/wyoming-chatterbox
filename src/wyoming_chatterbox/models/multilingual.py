"""Multilingual Chatterbox TTS backend."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from wyoming_chatterbox.config import Settings
from wyoming_chatterbox.metrics import count_voice_preparation_cache, observe_voice_preparation
from wyoming_chatterbox.models.base import ChatterboxBackend

logger = logging.getLogger(__name__)

# BCP-47 codes supported by ChatterboxMultilingualTTS (best-effort list).
_SUPPORTED_LANGUAGES: list[str] = [
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "nl",
    "ru",
    "ja",
    "ko",
    "zh",
]


class MultilingualBackend(ChatterboxBackend):
    """Wraps the multilingual ``ChatterboxMultilingualTTS`` model."""

    variant = "multilingual"
    sample_rate = 24000

    def __init__(self, device: str, settings: Settings) -> None:
        self._device = device
        self._settings = settings
        self._model: object | None = None
        self._prepared_audio_prompt_key: tuple[str, float] | None = None

    # -- lifecycle --------------------------------------------------------

    def load(self) -> None:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # lazy import

        self._model = ChatterboxMultilingualTTS.from_pretrained(device=self._device)

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
            "cfg_weight": self._settings.chatterbox_cfg_weight,
            "temperature": self._settings.chatterbox_temperature,
        }

    def _forward_audio_prompt_to_generate(self) -> bool:
        """Return True when the underlying model must receive ``audio_prompt_path``."""
        return False

    def generate(self, text: str, **kwargs: object) -> np.ndarray:
        self._ensure_loaded()
        gen_kwargs = self._build_generate_kwargs()
        language = kwargs.pop("language", None) or self._settings.chatterbox_default_language
        gen_kwargs["language_id"] = language
        gen_kwargs.update(kwargs)
        audio_prompt_path = gen_kwargs.get("audio_prompt_path")
        if audio_prompt_path:
            self._prepare_audio_prompt(str(audio_prompt_path), phase="request")
            if not self._forward_audio_prompt_to_generate():
                gen_kwargs.pop("audio_prompt_path", None)
        assert self._model is not None  # satisfied by _ensure_loaded
        audio = self._model.generate(text, **gen_kwargs)  # type: ignore[union-attr]
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        return np.asarray(audio, dtype=np.float32)

    # -- language ---------------------------------------------------------

    def supported_languages(self) -> list[str]:
        return list(_SUPPORTED_LANGUAGES)

    def supports_language(self, lang: str) -> bool:
        return lang.lower() in _SUPPORTED_LANGUAGES
