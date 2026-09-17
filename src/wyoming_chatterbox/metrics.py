"""Prometheus metrics helpers for Wyoming Chatterbox."""

from __future__ import annotations

import logging
from threading import Lock

from prometheus_client import Counter, Histogram, start_http_server

from wyoming_chatterbox.config import Settings

logger = logging.getLogger(__name__)

_METRICS_STARTED = False
_METRICS_LOCK = Lock()

_LATENCY_BUCKETS = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    20.0,
    30.0,
    60.0,
    120.0,
)

_COUNT_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
_BYTES_BUCKETS = (
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
    16777216,
)

SYNTHESIS_REQUESTS = Counter(
    "wyoming_chatterbox_synthesis_requests_total",
    "Total synthesis requests handled by Wyoming Chatterbox.",
    ("variant", "status"),
)
SYNTHESIS_DURATION = Histogram(
    "wyoming_chatterbox_synthesis_duration_seconds",
    "End-to-end synthesis request duration.",
    ("variant", "status"),
    buckets=_LATENCY_BUCKETS,
)
SYNTHESIS_FIRST_AUDIO = Histogram(
    "wyoming_chatterbox_synthesis_first_audio_seconds",
    "Time from synthesis request start to first audio chunk.",
    ("variant",),
    buckets=_LATENCY_BUCKETS,
)
SYNTHESIS_CHUNKS = Histogram(
    "wyoming_chatterbox_synthesis_chunks",
    "Audio chunks emitted per synthesis request.",
    ("variant",),
    buckets=_COUNT_BUCKETS,
)
SYNTHESIS_AUDIO_BYTES = Histogram(
    "wyoming_chatterbox_synthesis_audio_bytes",
    "Audio bytes emitted per synthesis request.",
    ("variant",),
    buckets=_BYTES_BUCKETS,
)
SEGMENT_DURATION = Histogram(
    "wyoming_chatterbox_segment_duration_seconds",
    "Per-segment synthesis duration.",
    ("variant",),
    buckets=_LATENCY_BUCKETS,
)
VOICE_PREPARATION = Histogram(
    "wyoming_chatterbox_voice_preparation_seconds",
    "Reference voice preparation duration.",
    ("variant", "phase", "status"),
    buckets=_LATENCY_BUCKETS,
)
VOICE_PREPARATION_CACHE = Counter(
    "wyoming_chatterbox_voice_preparation_cache_total",
    "Reference voice preparation cache events.",
    ("variant", "result"),
)


def start_metrics_server(settings: Settings) -> None:
    """Start the Prometheus exporter if enabled."""
    global _METRICS_STARTED
    if not settings.prometheus_enabled:
        return
    with _METRICS_LOCK:
        if _METRICS_STARTED:
            return
        try:
            start_http_server(port=settings.prometheus_port, addr=settings.prometheus_host)
        except Exception:  # noqa: BLE001 - metrics are optional
            logger.exception(
                "Failed to start Prometheus metrics server on %s:%s",
                settings.prometheus_host,
                settings.prometheus_port,
            )
            return
        _METRICS_STARTED = True
    logger.info(
        "Started Prometheus metrics server on %s:%s",
        settings.prometheus_host,
        settings.prometheus_port,
    )


def observe_synthesis_request(
    *,
    variant: str,
    status: str,
    duration_seconds: float,
    first_audio_seconds: float | None,
    chunk_count: int,
    audio_bytes: int,
) -> None:
    """Record end-to-end synthesis metrics."""
    SYNTHESIS_REQUESTS.labels(variant=variant, status=status).inc()
    SYNTHESIS_DURATION.labels(variant=variant, status=status).observe(duration_seconds)
    if status == "success":
        if first_audio_seconds is not None:
            SYNTHESIS_FIRST_AUDIO.labels(variant=variant).observe(first_audio_seconds)
        SYNTHESIS_CHUNKS.labels(variant=variant).observe(chunk_count)
        SYNTHESIS_AUDIO_BYTES.labels(variant=variant).observe(audio_bytes)


def observe_segment_duration(variant: str, duration_seconds: float) -> None:
    """Record segment synthesis duration."""
    SEGMENT_DURATION.labels(variant=variant).observe(duration_seconds)


def observe_voice_preparation(
    *,
    variant: str,
    phase: str,
    status: str,
    duration_seconds: float,
) -> None:
    """Record reference voice preparation duration."""
    VOICE_PREPARATION.labels(variant=variant, phase=phase, status=status).observe(duration_seconds)


def count_voice_preparation_cache(variant: str, result: str) -> None:
    """Record a reference voice preparation cache hit or miss."""
    VOICE_PREPARATION_CACHE.labels(variant=variant, result=result).inc()
