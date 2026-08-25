"""End-to-end server test using a real Wyoming TCP server and client."""

from __future__ import annotations

import asyncio
import socket

import numpy as np
import pytest
from conftest import FakeBackend
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info, SelectProgram
from wyoming.server import AsyncTcpServer
from wyoming.tts import Synthesize, SynthesizeVoice

from wyoming_chatterbox.config import Settings
from wyoming_chatterbox.server.handler import ChatterboxEventHandler
from wyoming_chatterbox.voices.manager import VoiceManager


def _free_port() -> int:
    """Return an OS-assigned free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server_settings(tmp_path) -> Settings:
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "alice.wav").write_bytes(b"RIFF")
    return Settings(
        chatterbox_variant="standard",
        chatterbox_device="cpu",
        chatterbox_preload=False,
        chatterbox_streaming_mode="segmented",
        chatterbox_voices_dir=str(voices),
        chatterbox_period_pause_ms=0,
        chatterbox_comma_pause_ms=0,
        wyoming_audio_chunk_ms=20,
    )


async def _start_server(settings):
    backend = FakeBackend()
    backend.load()
    backends = {"standard": backend}
    voice_manager = VoiceManager(settings.chatterbox_voices_dir)

    def factory(reader, writer):
        return ChatterboxEventHandler(reader, writer, backends, settings, voice_manager, "standard")

    port = _free_port()
    server = AsyncTcpServer(host="127.0.0.1", port=port)
    await server.start(factory)
    return server, port, backend


async def _shutdown(server):
    await server.stop()


async def test_describe_returns_info(server_settings):
    server, port, _ = await _start_server(server_settings)
    try:
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Describe().event())
            event = await asyncio.wait_for(client.read_event(), timeout=5)
            assert event is not None
            assert Info.is_type(event.type)
            info = Info.from_event(event)
            assert len(info.tts) == 1
            program = info.tts[0]
            assert program.name == "standard"
            voice_names = [v.name for v in program.voices]
            assert "alice" in voice_names
    finally:
        await _shutdown(server)


async def test_synthesize_streams_audio(server_settings):
    server, port, _ = await _start_server(server_settings)
    try:
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Synthesize(text="Hello world. This is a test.").event())
            start = await asyncio.wait_for(client.read_event(), timeout=5)
            assert AudioStart.is_type(start.type)
            audio_start = AudioStart.from_event(start)
            assert audio_start.rate == 24000
            assert audio_start.width == 2
            assert audio_start.channels == 1

            total = bytearray()
            while True:
                event = await asyncio.wait_for(client.read_event(), timeout=5)
                if AudioChunk.is_type(event.type):
                    total += AudioChunk.from_event(event).audio
                elif AudioStop.is_type(event.type):
                    break
                else:
                    raise AssertionError(f"Unexpected event {event.type}")

            assert len(total) > 0
            assert len(total) % 2 == 0
            samples = np.frombuffer(bytes(total), dtype="<i2")
            assert samples.dtype == np.int16
    finally:
        await _shutdown(server)


async def test_select_program_then_synthesize(server_settings):
    server, port, _ = await _start_server(server_settings)
    try:
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(SelectProgram(name="standard").event())
            await client.write_event(
                Synthesize(
                    text="Voiced request.",
                    voice=SynthesizeVoice(name="alice", language="en"),
                ).event()
            )
            start = await asyncio.wait_for(client.read_event(), timeout=5)
            assert AudioStart.is_type(start.type)
            saw_stop = False
            for _ in range(200):
                event = await asyncio.wait_for(client.read_event(), timeout=5)
                if AudioStop.is_type(event.type):
                    saw_stop = True
                    break
            assert saw_stop
    finally:
        await _shutdown(server)
