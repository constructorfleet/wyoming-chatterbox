"""Tests for Prometheus metrics setup."""

from __future__ import annotations

from wyoming_chatterbox import metrics
from wyoming_chatterbox.config import Settings


def test_start_metrics_server_disabled(monkeypatch):
    called = []
    monkeypatch.setattr(metrics, "start_http_server", lambda **kwargs: called.append(kwargs))
    monkeypatch.setattr(metrics, "_METRICS_STARTED", False)

    metrics.start_metrics_server(Settings(prometheus_enabled=False))

    assert called == []


def test_start_metrics_server_enabled(monkeypatch):
    called = []
    monkeypatch.setattr(metrics, "start_http_server", lambda **kwargs: called.append(kwargs))
    monkeypatch.setattr(metrics, "_METRICS_STARTED", False)

    metrics.start_metrics_server(
        Settings(prometheus_enabled=True, prometheus_host="127.0.0.1", prometheus_port=9200)
    )

    assert called == [{"addr": "127.0.0.1", "port": 9200}]


def test_start_metrics_server_only_once(monkeypatch):
    called = []
    monkeypatch.setattr(metrics, "start_http_server", lambda **kwargs: called.append(kwargs))
    monkeypatch.setattr(metrics, "_METRICS_STARTED", False)
    settings = Settings(prometheus_enabled=True)

    metrics.start_metrics_server(settings)
    metrics.start_metrics_server(settings)

    assert len(called) == 1
