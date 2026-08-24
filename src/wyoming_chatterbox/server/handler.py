"""Wyoming event handler for Chatterbox TTS."""

from __future__ import annotations

import asyncio
import logging

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
            logger.debug("Selected program %s", event.name)
        else:
            await self.write_event(Error(text=f"Unknown program: {event.name}").event())

    # -- synthesize -------------------------------------------------------

    async def _handle_synthesize(self, event: Synthesize) -> None:
        pipeline: SynthesisPipeline | None = None
        try:
            backend = self._backends[self._active_variant]
            if not backend.is_loaded:
                backend.load()

            pipeline = SynthesisPipeline(backend, self._settings, self._voice_manager)

            voice = None
            language = self._settings.chatterbox_default_language
            if event.voice is not None:
                voice = event.voice.name or None
                if event.voice.language:
                    language = event.voice.language
            if not voice and self._settings.chatterbox_default_voice:
                voice = self._settings.chatterbox_default_voice

            sample_rate = backend.sample_rate
            await self.write_event(AudioStart(rate=sample_rate, width=2, channels=1).event())

            async for chunk in pipeline.synthesize_stream(
                event.text, voice=voice, language=language
            ):
                if not chunk:
                    continue
                await self.write_event(
                    AudioChunk(audio=chunk, rate=sample_rate, width=2, channels=1).event()
                )

            await self.write_event(AudioStop().event())
        except Exception as exc:  # noqa: BLE001 - report all failures to client
            logger.exception("Synthesis error")
            await self.write_event(Error(text=str(exc)).event())
        finally:
            if pipeline is not None:
                pipeline.close()
