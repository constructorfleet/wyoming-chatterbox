"""Tests for the voice manager."""

from __future__ import annotations

import pytest
from wyoming_chatterbox.voices.manager import VoiceManager


@pytest.fixture
def voices_dir(tmp_path):
    d = tmp_path / "voices"
    d.mkdir()
    (d / "alice.wav").write_bytes(b"RIFF")
    (d / "bob.wav").write_bytes(b"RIFF")
    (d / "notes.txt").write_text("ignore")
    return d


def test_list_voices(voices_dir):
    mgr = VoiceManager(voices_dir)
    assert mgr.list_voices() == ["alice", "bob"]


def test_list_voices_missing_dir(tmp_path):
    mgr = VoiceManager(tmp_path / "nope")
    assert mgr.list_voices() == []


def test_get_voice_path(voices_dir):
    mgr = VoiceManager(voices_dir)
    path = mgr.get_voice_path("alice")
    assert path == voices_dir / "alice.wav"


def test_has_voice(voices_dir):
    mgr = VoiceManager(voices_dir)
    assert mgr.has_voice("alice") is True
    assert mgr.has_voice("missing") is False


def test_path_traversal_relative(voices_dir):
    mgr = VoiceManager(voices_dir)
    with pytest.raises(ValueError):
        mgr.get_voice_path("../etc/passwd")


def test_path_traversal_absolute(voices_dir):
    mgr = VoiceManager(voices_dir)
    with pytest.raises(ValueError):
        mgr.get_voice_path("/etc/passwd")


def test_path_traversal_backslash(voices_dir):
    mgr = VoiceManager(voices_dir)
    with pytest.raises(ValueError):
        mgr.get_voice_path("..\\secret")


def test_empty_name(voices_dir):
    mgr = VoiceManager(voices_dir)
    with pytest.raises(ValueError):
        mgr.get_voice_path("")


def test_missing_voice(voices_dir):
    mgr = VoiceManager(voices_dir)
    with pytest.raises(FileNotFoundError):
        mgr.get_voice_path("charlie")
