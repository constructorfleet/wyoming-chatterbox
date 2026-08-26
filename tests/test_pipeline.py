"""Tests for the streaming synthesis pipeline."""

from __future__ import annotations

import time

import numpy as np

from wyoming_chatterbox.config import Settings
from wyoming_chatterbox.models.base import ChatterboxBackend
from wyoming_chatterbox.synthesis.pipeline import SynthesisPipeline


class RecordingBackend(ChatterboxBackend):
    """Backend that records calls and returns a marker-valued block per segment."""

    variant = "standard"
    sample_rate = 24000

    def __init__(self, value_map=None, reverse_delay=False):
        self._value_map = value_map or {}
        self._reverse_delay = reverse_delay
        self.calls: list[dict] = []

    def load(self):
        pass

    def unload(self):
        pass

    @property
    def is_loaded(self):
        return True

    def generate(self, text: str, **kwargs) -> np.ndarray:
        self.calls.append({"text": text, **kwargs})
        value = self._value_map.get(text.strip(), 0.5)
        if self._reverse_delay:
            # Later (higher-value) segments finish first -> forces reordering.
            time.sleep(max(0.0, 0.15 - value))
        return np.full(480, value, dtype=np.float32)

    def supports_language(self, lang: str) -> bool:
        return True

    def supported_languages(self) -> list[str]:
        return ["en"]


def _pipeline_settings(**overrides) -> Settings:
    base = {
        "chatterbox_streaming_mode": "segmented",
        "chatterbox_segment_min_chars": 5,
        "chatterbox_segment_target_chars": 1000,
        "chatterbox_segment_max_chars": 1000,
        "chatterbox_prefetch_segments": 5,
        "chatterbox_synthesis_workers": 3,
        "chatterbox_synthesis_concurrency": 3,
        "chatterbox_period_pause_ms": 0,
        "chatterbox_comma_pause_ms": 0,
        "wyoming_audio_chunk_ms": 1000,
        "chatterbox_preload": False,
    }
    base.update(overrides)
    return Settings(**base)


async def _collect(pipeline: SynthesisPipeline, **kwargs) -> list[bytes]:
    return [chunk async for chunk in pipeline.synthesize_stream(**kwargs)]


def _first_samples(chunks: list[bytes]) -> list[int]:
    return [int(np.frombuffer(c, dtype="<i2")[0]) for c in chunks if c]


async def test_buffered_mode_returns_audio():
    backend = RecordingBackend()
    settings = _pipeline_settings(chatterbox_streaming_mode="buffered")
    pipeline = SynthesisPipeline(backend, settings)
    chunks = await _collect(pipeline, text="Hello world this is a test.")
    assert chunks
    assert b"".join(chunks)
    assert len(backend.calls) == 1
    pipeline.close()


async def test_off_mode_same_as_buffered():
    backend = RecordingBackend()
    settings = _pipeline_settings(chatterbox_streaming_mode="off")
    pipeline = SynthesisPipeline(backend, settings)
    chunks = await _collect(pipeline, text="One sentence. Two sentence.")
    assert len(backend.calls) == 1  # off mode does not segment
    assert chunks
    pipeline.close()


async def test_segmented_mode_in_order():
    sentences = [
        "First sentence here now.",
        "Second sentence here now.",
        "Third sentence here now.",
    ]
    value_map = {s: round((i + 1) * 0.1, 3) for i, s in enumerate(sentences)}
    backend = RecordingBackend(value_map=value_map)
    settings = _pipeline_settings()
    pipeline = SynthesisPipeline(backend, settings)
    chunks = await _collect(pipeline, text=" ".join(sentences))
    samples = _first_samples(chunks)
    # Ascending markers => produced in segment order.
    assert samples == sorted(samples)
    assert len(set(samples)) == 3
    pipeline.close()


async def test_out_of_order_completion_returns_in_order():
    sentences = [
        "First sentence here now.",
        "Second sentence here now.",
        "Third sentence here now.",
    ]
    value_map = {s: round((i + 1) * 0.1, 3) for i, s in enumerate(sentences)}
    # reverse_delay makes the last segment finish first.
    backend = RecordingBackend(value_map=value_map, reverse_delay=True)
    settings = _pipeline_settings()
    pipeline = SynthesisPipeline(backend, settings)
    chunks = await _collect(pipeline, text=" ".join(sentences))
    samples = _first_samples(chunks)
    assert samples == sorted(samples)  # still in order despite reverse completion
    pipeline.close()


async def test_deterministic_per_segment_seeds():
    sentences = [
        "First sentence here now.",
        "Second sentence here now.",
        "Third sentence here now.",
    ]
    backend = RecordingBackend()
    settings = _pipeline_settings()
    pipeline = SynthesisPipeline(backend, settings)
    await _collect(pipeline, text=" ".join(sentences), seed=100)
    seeds = sorted(c["seed"] for c in backend.calls)
    assert seeds == [100, 101, 102]
    pipeline.close()


async def test_segmented_empty_text():
    backend = RecordingBackend()
    settings = _pipeline_settings()
    pipeline = SynthesisPipeline(backend, settings)
    chunks = await _collect(pipeline, text="   ")
    assert chunks == []
    pipeline.close()


async def test_pause_inserted_between_segments():
    sentences = ["First sentence here now.", "Second sentence here now."]
    backend = RecordingBackend()
    settings = _pipeline_settings(chatterbox_period_pause_ms=100)
    pipeline = SynthesisPipeline(backend, settings)
    chunks = await _collect(pipeline, text=" ".join(sentences))
    # Expect at least one silent chunk between the two segments.
    assert any(set(c) == {0} for c in chunks)
    pipeline.close()
