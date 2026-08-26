"""Audio helpers: PCM conversion, chunking, silence, crossfade, trimming."""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2  # int16


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert a float32 array in [-1, 1] to signed 16-bit little-endian PCM."""
    if audio.size == 0:
        return b""
    clipped = np.clip(audio.astype(np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def chunk_audio(pcm_bytes: bytes, chunk_ms: int, sample_rate: int) -> list[bytes]:
    """Split raw PCM16 bytes into chunks of ``chunk_ms`` milliseconds."""
    if not pcm_bytes:
        return []
    samples_per_chunk = max(1, int(sample_rate * chunk_ms / 1000))
    chunk_size = samples_per_chunk * BYTES_PER_SAMPLE
    return [pcm_bytes[i : i + chunk_size] for i in range(0, len(pcm_bytes), chunk_size)]


def make_silence(duration_ms: int, sample_rate: int) -> bytes:
    """Return ``duration_ms`` of silence as PCM16 bytes."""
    if duration_ms <= 0:
        return b""
    samples = int(sample_rate * duration_ms / 1000)
    return b"\x00" * (samples * BYTES_PER_SAMPLE)


def crossfade(audio1: np.ndarray, audio2: np.ndarray, fade_ms: int, sample_rate: int) -> np.ndarray:
    """Crossfade two float32 arrays, falling back to concatenation when too short."""
    fade_samples = int(sample_rate * fade_ms / 1000)
    if fade_samples <= 0 or len(audio1) < fade_samples or len(audio2) < fade_samples:
        return np.concatenate([audio1, audio2])
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    overlap = audio1[-fade_samples:] * fade_out + audio2[:fade_samples] * fade_in
    return np.concatenate([audio1[:-fade_samples], overlap, audio2[fade_samples:]])


def trim_silence(
    audio: np.ndarray, threshold: float = 0.01, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """Trim leading and trailing near-silence based on a simple amplitude threshold."""
    if len(audio) == 0:
        return audio
    abs_audio = np.abs(audio)
    mask = abs_audio > threshold
    if not np.any(mask):
        return audio
    first = int(np.argmax(mask))
    last = len(audio) - int(np.argmax(mask[::-1]))
    return audio[first:last]
