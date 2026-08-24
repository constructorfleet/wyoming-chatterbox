"""Streaming synthesis pipeline with ordered, bounded parallel workers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from wyoming_chatterbox.audio.processing import (
    chunk_audio,
    float32_to_pcm16,
    make_silence,
)
from wyoming_chatterbox.config import Settings
from wyoming_chatterbox.models.base import ChatterboxBackend
from wyoming_chatterbox.segmentation.segmenter import TextSegmenter
from wyoming_chatterbox.voices.manager import VoiceManager

logger = logging.getLogger(__name__)

_SENTENCE_ENDINGS = (".", "!", "?")
_CLAUSE_ENDINGS = (",", ";", ":")


def _apply_seed(seed: int) -> None:
    """Best-effort deterministic seeding.

    Seeds torch's global RNG (safe because the backend lock serialises inference).
    The global numpy state is intentionally left untouched to avoid races between
    concurrent worker threads; callers that need numpy randomness should use a
    local ``np.random.default_rng(seed)`` instance.
    """
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - no GPU in tests
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch missing
        pass


class SynthesisPipeline:
    """Convert text to a stream of PCM16 chunks using a Chatterbox backend."""

    def __init__(
        self,
        backend: ChatterboxBackend,
        settings: Settings,
        voice_manager: VoiceManager | None = None,
    ) -> None:
        self._backend = backend
        self._settings = settings
        self._voice_manager = voice_manager
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, settings.chatterbox_synthesis_workers)
        )

    def close(self) -> None:
        """Shut down the worker thread pool."""
        self._executor.shutdown(wait=False)

    # -- public API -------------------------------------------------------

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
        seed: int | None = None,
    ) -> AsyncIterator[bytes]:
        """Yield PCM16 byte chunks for ``text``."""
        base_seed = seed if seed is not None else self._settings.chatterbox_seed
        audio_prompt_path = self._resolve_voice(voice)
        gen_kwargs = {
            "language": language,
            "audio_prompt_path": audio_prompt_path,
        }

        mode = self._settings.chatterbox_streaming_mode
        if mode == "segmented":
            async for chunk in self._segmented_stream(text, base_seed, gen_kwargs):
                yield chunk
        else:
            async for chunk in self._buffered_stream(text, base_seed, gen_kwargs):
                yield chunk

    # -- buffered / off mode ---------------------------------------------

    async def _buffered_stream(
        self, text: str, base_seed: int | None, gen_kwargs: dict
    ) -> AsyncIterator[bytes]:
        _, audio = await self._run_in_executor(0, text, base_seed, gen_kwargs)
        for chunk in self._audio_to_chunks(audio):
            yield chunk

    # -- segmented mode ---------------------------------------------------

    async def _segmented_stream(
        self, text: str, base_seed: int | None, gen_kwargs: dict
    ) -> AsyncIterator[bytes]:
        segmenter = TextSegmenter(
            min_chars=self._settings.chatterbox_segment_min_chars,
            target_chars=self._settings.chatterbox_segment_target_chars,
            max_chars=self._settings.chatterbox_segment_max_chars,
        )
        segments = segmenter.feed(text)
        segments += segmenter.flush()
        if not segments:
            return

        max_in_flight = max(1, self._settings.chatterbox_prefetch_segments)
        semaphore = asyncio.Semaphore(max(1, self._settings.chatterbox_synthesis_concurrency))

        async def worker(seq: int, segment_text: str) -> tuple[int, np.ndarray]:
            async with semaphore:
                return await self._run_in_executor(seq, segment_text, base_seed, gen_kwargs)

        tasks: dict[int, asyncio.Task] = {}
        next_to_yield = 0
        next_to_schedule = 0
        total = len(segments)

        try:
            while next_to_yield < total:
                while (
                    next_to_schedule < total and (next_to_schedule - next_to_yield) < max_in_flight
                ):
                    tasks[next_to_schedule] = asyncio.ensure_future(
                        worker(next_to_schedule, segments[next_to_schedule])
                    )
                    next_to_schedule += 1

                _, audio = await tasks.pop(next_to_yield)
                for chunk in self._audio_to_chunks(audio):
                    yield chunk

                if next_to_yield < total - 1:
                    pause = self._pause_after(segments[next_to_yield])
                    if pause:
                        yield pause
                next_to_yield += 1
        finally:
            for task in tasks.values():
                task.cancel()

    # -- helpers ----------------------------------------------------------

    async def _run_in_executor(
        self, seq: int, text: str, base_seed: int | None, gen_kwargs: dict
    ) -> tuple[int, np.ndarray]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._synthesize_segment, seq, text, base_seed, gen_kwargs
        )

    def _synthesize_segment(
        self, seq: int, text: str, base_seed: int | None, gen_kwargs: dict
    ) -> tuple[int, np.ndarray]:
        """Synthesize a single segment (runs in the thread pool)."""
        kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        seg_seed = None
        if base_seed is not None:
            seg_seed = base_seed + seq
            _apply_seed(seg_seed)
            kwargs["seed"] = seg_seed
        audio = self._backend.generate(text, **kwargs)
        return seq, np.asarray(audio, dtype=np.float32).reshape(-1)

    def _audio_to_chunks(self, audio: np.ndarray) -> list[bytes]:
        pcm = float32_to_pcm16(audio)
        return chunk_audio(
            pcm,
            self._settings.wyoming_audio_chunk_ms,
            self._backend.sample_rate,
        )

    def _pause_after(self, segment: str) -> bytes:
        stripped = segment.rstrip()
        if not stripped:
            return b""
        last = stripped[-1]
        duration = 0
        if last in _SENTENCE_ENDINGS:
            duration = self._settings.chatterbox_period_pause_ms
        elif last in _CLAUSE_ENDINGS:
            duration = self._settings.chatterbox_comma_pause_ms
        if duration <= 0:
            return b""
        return make_silence(duration, self._backend.sample_rate)

    def _resolve_voice(self, voice: str | None) -> str | None:
        if not voice or self._voice_manager is None:
            return None
        try:
            return str(self._voice_manager.get_voice_path(voice))
        except (ValueError, FileNotFoundError):
            logger.warning("Reference voice %r not found; using default", voice)
            return None
