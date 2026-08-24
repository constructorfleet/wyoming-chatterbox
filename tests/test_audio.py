"""Tests for audio processing helpers."""

from __future__ import annotations

import numpy as np

from wyoming_chatterbox.audio.processing import (
    chunk_audio,
    crossfade,
    float32_to_pcm16,
    make_silence,
    trim_silence,
)


def test_float32_to_pcm16_known_values():
    audio = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
    pcm = float32_to_pcm16(audio)
    samples = np.frombuffer(pcm, dtype="<i2")
    assert samples[0] == 0
    assert samples[1] == 32767
    assert samples[2] == -32767
    assert abs(int(samples[3]) - 16383) <= 1


def test_float32_to_pcm16_clips():
    audio = np.array([2.0, -2.0], dtype=np.float32)
    samples = np.frombuffer(float32_to_pcm16(audio), dtype="<i2")
    assert samples[0] == 32767
    assert samples[1] == -32767


def test_float32_to_pcm16_empty():
    assert float32_to_pcm16(np.array([], dtype=np.float32)) == b""


def test_chunk_audio_sizes():
    # 24000 Hz, 20ms -> 480 samples -> 960 bytes per chunk.
    pcm = b"\x00\x00" * 2400  # 2400 samples = 100 ms
    chunks = chunk_audio(pcm, chunk_ms=20, sample_rate=24000)
    assert len(chunks) == 5
    assert all(len(c) == 960 for c in chunks)


def test_chunk_audio_remainder():
    pcm = b"\x00\x00" * 500  # 500 samples
    chunks = chunk_audio(pcm, chunk_ms=20, sample_rate=24000)  # 480 samples/chunk
    assert len(chunks) == 2
    assert len(chunks[0]) == 960
    assert len(chunks[1]) == 40


def test_chunk_audio_empty():
    assert chunk_audio(b"", 20, 24000) == []


def test_make_silence_length():
    silence = make_silence(100, 24000)  # 2400 samples * 2 bytes
    assert len(silence) == 4800
    assert silence == b"\x00" * 4800


def test_make_silence_zero():
    assert make_silence(0, 24000) == b""


def test_crossfade_output_length():
    a = np.ones(1000, dtype=np.float32)
    b = np.ones(1000, dtype=np.float32)
    out = crossfade(a, b, fade_ms=10, sample_rate=24000)  # 240 fade samples
    assert len(out) == 2000 - 240


def test_crossfade_too_short_concatenates():
    a = np.ones(10, dtype=np.float32)
    b = np.ones(10, dtype=np.float32)
    out = crossfade(a, b, fade_ms=10, sample_rate=24000)
    assert len(out) == 20


def test_trim_silence():
    audio = np.concatenate([np.zeros(100), np.ones(50) * 0.5, np.zeros(100)]).astype(np.float32)
    trimmed = trim_silence(audio, threshold=0.01)
    assert len(trimmed) == 50


def test_trim_silence_all_silent():
    audio = np.zeros(100, dtype=np.float32)
    assert len(trim_silence(audio)) == 100
