"""Tests for model backends and the factory."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
from wyoming_chatterbox.models.factory import create_backend, resolve_device
from wyoming_chatterbox.models.multilingual import MultilingualBackend
from wyoming_chatterbox.models.nano import NanoBackend
from wyoming_chatterbox.models.standard import StandardBackend
from wyoming_chatterbox.models.turbo import TurboBackend

from wyoming_chatterbox.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(chatterbox_device="cpu", chatterbox_preload=False)


@pytest.mark.parametrize(
    "variant,expected",
    [
        ("standard", StandardBackend),
        ("multilingual", MultilingualBackend),
        ("turbo", TurboBackend),
        ("nano", NanoBackend),
    ],
)
def test_factory_creates_correct_type(variant, expected, settings):
    backend = create_backend(variant, "cpu", settings)
    assert isinstance(backend, expected)
    assert backend.variant == variant


def test_factory_unknown_variant(settings):
    with pytest.raises(ValueError):
        create_backend("bogus", "cpu", settings)


def test_resolve_device_auto_cuda(monkeypatch):
    monkeypatch.setattr("wyoming_chatterbox.models.factory._cuda_available", lambda: True)
    monkeypatch.setattr("wyoming_chatterbox.models.factory._mps_available", lambda: False)
    assert resolve_device("auto") == "cuda"


def test_resolve_device_auto_mps(monkeypatch):
    monkeypatch.setattr("wyoming_chatterbox.models.factory._cuda_available", lambda: False)
    monkeypatch.setattr("wyoming_chatterbox.models.factory._mps_available", lambda: True)
    assert resolve_device("auto") == "mps"


def test_resolve_device_auto_cpu(monkeypatch):
    monkeypatch.setattr("wyoming_chatterbox.models.factory._cuda_available", lambda: False)
    monkeypatch.setattr("wyoming_chatterbox.models.factory._mps_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_resolve_device_explicit_cuda_unavailable(monkeypatch):
    monkeypatch.setattr("wyoming_chatterbox.models.factory._cuda_available", lambda: False)
    with pytest.raises(RuntimeError):
        resolve_device("cuda")


def test_resolve_device_explicit_mps_unavailable(monkeypatch):
    monkeypatch.setattr("wyoming_chatterbox.models.factory._mps_available", lambda: False)
    with pytest.raises(RuntimeError):
        resolve_device("mps")


def _install_fake_chatterbox(monkeypatch, model):
    """Install a fake chatterbox.tts module so load() can import it."""
    tts_module = types.ModuleType("chatterbox.tts")
    tts_module.ChatterboxTTS = MagicMock()
    tts_module.ChatterboxTTS.from_pretrained.return_value = model
    pkg = types.ModuleType("chatterbox")
    monkeypatch.setitem(sys.modules, "chatterbox", pkg)
    monkeypatch.setitem(sys.modules, "chatterbox.tts", tts_module)
    return tts_module


def test_lazy_load(monkeypatch, settings):
    model = MagicMock()
    model.sr = 24000
    model.generate.return_value = np.zeros(2400, dtype=np.float32)
    tts_module = _install_fake_chatterbox(monkeypatch, model)

    backend = StandardBackend("cpu", settings)
    assert backend.is_loaded is False
    # Not loaded until generate() or load() is called.
    tts_module.ChatterboxTTS.from_pretrained.assert_not_called()

    audio = backend.generate("hello")
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert backend.is_loaded is True


def test_load_once(monkeypatch, settings):
    model = MagicMock()
    model.sr = 24000
    model.generate.return_value = np.zeros(2400, dtype=np.float32)
    tts_module = _install_fake_chatterbox(monkeypatch, model)

    backend = StandardBackend("cpu", settings)
    backend.generate("first")
    backend.generate("second")
    backend.generate("third")
    tts_module.ChatterboxTTS.from_pretrained.assert_called_once()


def test_generate_converts_torch_tensor(monkeypatch, settings):
    class FakeTensor:
        def __init__(self, arr):
            self._arr = arr

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    model = MagicMock()
    model.sr = 24000
    model.generate.return_value = FakeTensor(np.zeros(1200, dtype=np.float32))
    _install_fake_chatterbox(monkeypatch, model)

    backend = StandardBackend("cpu", settings)
    audio = backend.generate("hi")
    assert audio.dtype == np.float32
    assert audio.shape == (1200,)


def test_multilingual_languages(settings):
    backend = MultilingualBackend("cpu", settings)
    langs = backend.supported_languages()
    assert "en" in langs
    assert "fr" in langs
    assert backend.supports_language("ES") is True
    assert backend.supports_language("xx") is False


def test_standard_supports_english_only(settings):
    backend = StandardBackend("cpu", settings)
    assert backend.supported_languages() == ["en"]
    assert backend.supports_language("en") is True
    assert backend.supports_language("fr") is False


def test_unload(monkeypatch, settings):
    model = MagicMock()
    model.sr = 24000
    _install_fake_chatterbox(monkeypatch, model)
    backend = StandardBackend("cpu", settings)
    backend.load()
    assert backend.is_loaded
    backend.unload()
    assert not backend.is_loaded


def _install_fake_multilingual(monkeypatch, model):
    module = types.ModuleType("chatterbox.mtl_tts")
    module.ChatterboxMultilingualTTS = MagicMock()
    module.ChatterboxMultilingualTTS.from_pretrained.return_value = model
    monkeypatch.setitem(sys.modules, "chatterbox", types.ModuleType("chatterbox"))
    monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts", module)
    return module


def test_multilingual_generate_passes_language(monkeypatch, settings):
    model = MagicMock()
    model.sr = 24000
    model.generate.return_value = np.zeros(2400, dtype=np.float32)
    _install_fake_multilingual(monkeypatch, model)

    backend = MultilingualBackend("cpu", settings)
    backend.generate("bonjour", language="fr")
    _, kwargs = model.generate.call_args
    assert kwargs["language_id"] == "fr"


def test_multilingual_generate_default_language(monkeypatch):
    settings = Settings(
        chatterbox_device="cpu",
        chatterbox_preload=False,
        chatterbox_default_language="de",
    )
    model = MagicMock()
    model.sr = 24000
    model.generate.return_value = np.zeros(2400, dtype=np.float32)
    _install_fake_multilingual(monkeypatch, model)

    backend = MultilingualBackend("cpu", settings)
    backend.generate("hallo")
    _, kwargs = model.generate.call_args
    assert kwargs["language_id"] == "de"


def test_turbo_caps_cfg_weight(settings):
    backend = TurboBackend("cpu", settings)
    params = backend._build_generate_kwargs()
    assert params["cfg_weight"] <= 0.3


def test_generate_passes_voice_prompt(monkeypatch, settings):
    model = MagicMock()
    model.sr = 24000
    model.generate.return_value = np.zeros(2400, dtype=np.float32)
    _install_fake_chatterbox(monkeypatch, model)

    backend = StandardBackend("cpu", settings)
    backend.generate("hi", audio_prompt_path="/voices/alice.wav")
    _, kwargs = model.generate.call_args
    assert kwargs["audio_prompt_path"] == "/voices/alice.wav"
