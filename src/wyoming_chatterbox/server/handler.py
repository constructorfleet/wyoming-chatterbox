"""Wyoming event handler for Chatterbox TTS."""

from __future__ import annotations

import asyncio
import logging
import time

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import (
    Attribution,
    Describe,
    Info,
    SelectProgram,
    TtsProgram,
    TtsVoice,
)
from wyoming.server import AsyncEventHandler
from wyoming.tts import Synthesize

from wyoming_chatterbox.config import Settings
from wyoming_chatterbox.metrics import observe_synthesis_request
from wyoming_chatterbox.models.base import ChatterboxBackend
from wyoming_chatterbox.synthesis.pipeline import SynthesisPipeline
from wyoming_chatterbox.voices.manager import VoiceManager

logger = logging.getLogger(__name__)

_ATTRIBUTION = Attribution(
    name="Resemble AI",
    url="https://github.com/resemble-ai/chatterbox",
)


class ChatterboxEventHandler(AsyncEventHandler):
    """Handle Wyoming protocol events for one client connection."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        backends: dict[str, ChatterboxBackend],
        settings: Settings,
        voice_manager: VoiceManager,
        default_variant: str,
    ) -> None:
        super().__init__(reader, writer)
        self._backends = backends
        self._settings = settings
        self._voice_manager = voice_manager
        self._active_variant = default_variant
        # Create one pipeline per connection — the pipeline's ThreadPoolExecutor is
        # reused across requests on the same connection instead of being torn down
        # after every synthesis call.
        self._pipeline = SynthesisPipeline(backends[default_variant], settings, voice_manager)

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self._handle_describe()
            return True
        if SelectProgram.is_type(event.type):
            await self._handle_select_program(SelectProgram.from_event(event))
            return True
        if Synthesize.is_type(event.type):
            await self._handle_synthesize(Synthesize.from_event(event))
            return True
        return True

    # -- describe ---------------------------------------------------------

    async def _handle_describe(self) -> None:
        await self.write_event(self._build_info().event())

    def _build_info(self) -> Info:
        voice_names = self._voice_manager.list_voices()
        programs: list[TtsProgram] = []
        for variant, backend in self._backends.items():
            languages = backend.supported_languages()
            voices = [
                TtsVoice(
                    name=name,
                    description=f"Reference voice {name}",
                    attribution=_ATTRIBUTION,
                    installed=True,
                    version=None,
                    languages=list(languages),
                )
                for name in voice_names
            ]
            programs.append(
                TtsProgram(
                    name=variant,
                    description=f"Chatterbox TTS ({variant})",
                    attribution=_ATTRIBUTION,
                    installed=backend.is_loaded,
                    version=None,
                    voices=voices,
                    supports_synthesize_streaming=True,
                )
            )
        return Info(tts=programs)

    # -- select program ---------------------------------------------------

    async def _handle_select_program(self, event: SelectProgram) -> None:
        if event.name in self._backends:
            self._active_variant = event.name
            # Re-point the shared pipeline to the newly selected backend.
            self._pipeline.close()
            self._pipeline = SynthesisPipeline(
                self._backends[event.name], self._settings, self._voice_manager
            )
            logger.debug("Selected program %s", event.name)
        else:
            await self.write_event(Error(text=f"Unknown program: {event.name}").event())

    # -- synthesize -------------------------------------------------------

    async def _handle_synthesize(self, event: Synthesize) -> None:
        request_start = time.perf_counter()
        first_audio_at: float | None = None
        chunk_count = 0
        audio_bytes = 0
        variant = self._active_variant
        voice = None
        language = self._settings.chatterbox_default_language
        try:
            backend = self._backends[self._active_variant]
            if not backend.is_loaded:
                backend.load()

            if event.voice is not None:
                voice = event.voice.name or None
                if event.voice.language:
                    language = event.voice.language
            if not voice and self._settings.chatterbox_default_voice:
                voice = self._settings.chatterbox_default_voice

            logger.info(
                "Starting synthesis variant=%s text_chars=%d voice=%s language=%s",
                variant,
                len(event.text),
                voice or "-",
                language,
            )

            sample_rate = backend.sample_rate
            await self.write_event(AudioStart(rate=sample_rate, width=2, channels=1).event())

            async for chunk in self._pipeline.synthesize_stream(
                event.text, voice=voice, language=language
            ):
                if not chunk:
                    continue
                if first_audio_at is None:
                    first_audio_at = time.perf_counter()
                chunk_count += 1
                audio_bytes += len(chunk)
                await self.write_event(
                    AudioChunk(audio=chunk, rate=sample_rate, width=2, channels=1).event()
                )

            await self.write_event(AudioStop().event())
            duration = time.perf_counter() - request_start
            first_audio_seconds = None
            if first_audio_at is not None:
                first_audio_seconds = first_audio_at - request_start
            observe_synthesis_request(
                variant=variant,
                status="success",
                duration_seconds=duration,
                first_audio_seconds=first_audio_seconds,
                chunk_count=chunk_count,
                audio_bytes=audio_bytes,
            )
            logger.info(
                "Completed synthesis variant=%s text_chars=%d voice=%s language=%s chunks=%d "
                "audio_bytes=%d first_audio_ms=%s total_ms=%.1f",
                variant,
                len(event.text),
                voice or "-",
                language,
                chunk_count,
                audio_bytes,
                f"{first_audio_seconds * 1000.0:.1f}" if first_audio_seconds is not None else "n/a",
                duration * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001 - report all failures to client
            duration = time.perf_counter() - request_start
            observe_synthesis_request(
                variant=variant,
                status="error",
                duration_seconds=duration,
                first_audio_seconds=None,
                chunk_count=chunk_count,
                audio_bytes=audio_bytes,
            )
            logger.exception(
                "Synthesis error variant=%s text_chars=%d voice=%s language=%s total_ms=%.1f",
                variant,
                len(event.text),
                voice or "-",
                language,
                duration * 1000.0,
            )
            await self.write_event(Error(text=str(exc)).event())
