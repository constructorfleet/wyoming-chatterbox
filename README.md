# wyoming-chatterbox

A production-quality [Wyoming protocol](https://github.com/rhasspy/wyoming) text-to-speech
server for [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox), designed for
use with [Home Assistant](https://www.home-assistant.io/).

## Features

- **Four model variants** — `standard`, `multilingual`, `turbo`, and `nano`, served
  individually or together as selectable Wyoming programs.
- **Device selection** — `auto`, `cpu`, `cuda`, or `mps`. Explicit accelerators fail fast if
  unavailable instead of silently falling back.
- **Segmented streaming synthesis** — incremental text segmentation with a bounded,
  ordered, parallel synthesis pipeline for low time-to-first-audio.
- **Named reference voices** — drop `*.wav` files into the voices directory and select them
  by name (with path-traversal protection).
- **Multilingual** — the multilingual backend exposes per-request language selection.
- **Fully mockable** — the Chatterbox dependency is imported lazily so the package (and its
  test suite) runs without downloading any models.

## Installation

```bash
pip install wyoming-chatterbox
# plus the runtime model dependencies:
pip install "wyoming-chatterbox[torch,chatterbox]"
```

## Usage

Configuration is entirely environment-variable driven (see [`.env.example`](.env.example)):

```bash
CHATTERBOX_VARIANT=multilingual CHATTERBOX_DEVICE=auto wyoming-chatterbox
```

The server listens on `tcp://0.0.0.0:10200` by default.

### Docker

```bash
docker compose up -d          # CPU image
```

Build the CUDA image by setting `target: cuda` in `compose.yaml` and enabling the GPU
`deploy` block. Published images are available at
`ghcr.io/wyoming-chatterbox/wyoming-chatterbox` with `-cpu` / `-cuda` tags.

### Home Assistant

Add the Wyoming integration and point it at `host:10200`. Each configured variant appears as
a selectable TTS program; each `*.wav` in the voices directory appears as a voice.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `WYOMING_HOST` | `0.0.0.0` | Bind host |
| `WYOMING_PORT` | `10200` | Bind port |
| `WYOMING_AUDIO_CHUNK_MS` | `20` | Output audio chunk size |
| `CHATTERBOX_VARIANT` | `multilingual` | Variant: `standard`/`multilingual`/`turbo`/`nano` |
| `CHATTERBOX_VARIANTS` | _(empty)_ | Comma-separated list to serve multiple variants |
| `CHATTERBOX_DEVICE` | `auto` | `auto`/`cpu`/`cuda`/`mps` |
| `CHATTERBOX_PRELOAD` | `true` | Load models at startup |
| `CHATTERBOX_CACHE_DIR` | `/models` | Model cache directory |
| `CHATTERBOX_VOICES_DIR` | `/voices` | Reference voice directory |
| `CHATTERBOX_DEFAULT_VOICE` | _(empty)_ | Default reference voice name |
| `CHATTERBOX_DEFAULT_LANGUAGE` | `en` | Default language |
| `CHATTERBOX_STREAMING_MODE` | `segmented` | `off`/`buffered`/`segmented` |
| `CHATTERBOX_SEGMENT_MIN_CHARS` | `40` | Minimum segment length |
| `CHATTERBOX_SEGMENT_TARGET_CHARS` | `160` | Target segment length |
| `CHATTERBOX_SEGMENT_MAX_CHARS` | `280` | Maximum segment length |
| `CHATTERBOX_PREFETCH_SEGMENTS` | `2` | In-flight segment prefetch depth |
| `CHATTERBOX_SYNTHESIS_WORKERS` | `1` | Thread pool worker count |
| `CHATTERBOX_SYNTHESIS_CONCURRENCY` | `2` | Max concurrent segment syntheses |
| `CHATTERBOX_SEED` | _(none)_ | Base RNG seed (segment `n` uses `seed + n`) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `text` | `text` or `json` |

See [`.env.example`](.env.example) for the full list including generation and audio-boundary
parameters.

## Streaming modes

- **`off` / `buffered`** — synthesize the whole utterance, then stream it out in chunks.
- **`segmented`** — split text into sentence-like segments, synthesize them with a bounded
  parallel worker pool, and emit audio strictly in order with short pauses at sentence and
  clause boundaries. This minimizes time-to-first-audio for longer responses.

## Development

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pyright src/
pytest
```

The test suite mocks Chatterbox entirely and never downloads models.

## License

MIT
