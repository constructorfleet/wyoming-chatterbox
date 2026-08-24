"""Wyoming TCP server setup for Chatterbox."""

from __future__ import annotations

import asyncio
import logging
import signal

from wyoming.server import AsyncTcpServer

from wyoming_chatterbox.config import Settings
from wyoming_chatterbox.models.factory import create_backend, resolve_device
from wyoming_chatterbox.server.handler import ChatterboxEventHandler
from wyoming_chatterbox.voices.manager import VoiceManager

logger = logging.getLogger(__name__)


async def run_server(settings: Settings) -> None:
    """Build backends and run the Wyoming TCP server until stopped."""
    device = resolve_device(settings.chatterbox_device)
    logger.info("Using device: %s", device)

    variants = settings.active_variants
    backends = {v: create_backend(v, device, settings) for v in variants}

    if settings.chatterbox_preload:
        for variant, backend in backends.items():
            logger.info("Preloading %s model...", variant)
            backend.load()

    voice_manager = VoiceManager(settings.chatterbox_voices_dir)
    default_variant = variants[0]

    server = AsyncTcpServer(host=settings.wyoming_host, port=settings.wyoming_port)

    def handler_factory(reader, writer) -> ChatterboxEventHandler:
        return ChatterboxEventHandler(
            reader,
            writer,
            backends,
            settings,
            voice_manager,
            default_variant,
        )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except (NotImplementedError, ValueError):  # pragma: no cover - platform
            pass

    logger.info(
        "Starting Wyoming Chatterbox server on %s:%s (variants: %s)",
        settings.wyoming_host,
        settings.wyoming_port,
        ", ".join(variants),
    )

    server_task = asyncio.ensure_future(server.run(handler_factory))
    stop_task = asyncio.ensure_future(stop_event.wait())
    try:
        done, _ = await asyncio.wait({server_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        if server_task in done:
            # Propagate any server error.
            server_task.result()
    finally:
        await server.stop()
        for task in (server_task, stop_task):
            if not task.done():
                task.cancel()
