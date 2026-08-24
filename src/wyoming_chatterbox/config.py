"""Application configuration via pydantic-settings."""

from __future__ import annotations

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

VALID_DEVICES = {"auto", "cpu", "cuda", "mps"}
VALID_VARIANTS = {"standard", "multilingual", "turbo", "nano"}
VALID_STREAMING_MODES = {"off", "buffered", "segmented"}


class Settings(BaseSettings):
    """Runtime configuration for the Wyoming Chatterbox server."""

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    # Wyoming
    wyoming_host: str = "0.0.0.0"
    wyoming_port: int = 10200
    wyoming_audio_chunk_ms: int = 20

    # Chatterbox
    chatterbox_variant: str = "multilingual"
    chatterbox_variants: str = ""
    chatterbox_device: str = "auto"
    chatterbox_preload: bool = True
    chatterbox_model: str = ""
    chatterbox_cache_dir: str = "/models"
    chatterbox_voices_dir: str = "/voices"
    chatterbox_default_voice: str = ""
    chatterbox_default_language: str = "en"

    # Streaming
    chatterbox_streaming_mode: str = "segmented"
    chatterbox_segment_min_chars: int = 40
    chatterbox_segment_target_chars: int = 160
    chatterbox_segment_max_chars: int = 280
    chatterbox_segment_flush_ms: int = 250
    chatterbox_prefetch_segments: int = 2
    chatterbox_synthesis_workers: int = 1
    chatterbox_synthesis_concurrency: int = 2

    # Generation
    chatterbox_exaggeration: float = 0.5
    chatterbox_cfg_weight: float = 0.5
    chatterbox_temperature: float = 0.8
    chatterbox_top_p: float = 1.0
    chatterbox_min_p: float = 0.05
    chatterbox_top_k: int = 1000
    chatterbox_repetition_penalty: float = 1.2
    chatterbox_seed: int | None = None

    # Audio boundary
    chatterbox_period_pause_ms: int = 120
    chatterbox_comma_pause_ms: int = 50
    chatterbox_crossfade_ms: int = 10

    # Logging
    log_level: str = "INFO"
    log_format: str = "text"
    hf_token: str = ""

    @field_validator("chatterbox_device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in VALID_DEVICES:
            raise ValueError(f"Invalid device {value!r}; must be one of {sorted(VALID_DEVICES)}")
        return normalized

    @field_validator("chatterbox_variant")
    @classmethod
    def _validate_variant(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in VALID_VARIANTS:
            raise ValueError(f"Invalid variant {value!r}; must be one of {sorted(VALID_VARIANTS)}")
        return normalized

    @field_validator("chatterbox_streaming_mode")
    @classmethod
    def _validate_streaming_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in VALID_STREAMING_MODES:
            raise ValueError(
                f"Invalid streaming mode {value!r}; must be one of {sorted(VALID_STREAMING_MODES)}"
            )
        return normalized

    @field_validator("log_format")
    @classmethod
    def _validate_log_format(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"text", "json"}:
            raise ValueError(f"Invalid log format {value!r}; must be 'text' or 'json'")
        return normalized

    @field_validator(
        "wyoming_port",
        "wyoming_audio_chunk_ms",
        "chatterbox_segment_min_chars",
        "chatterbox_segment_target_chars",
        "chatterbox_segment_max_chars",
        "chatterbox_segment_flush_ms",
        "chatterbox_prefetch_segments",
        "chatterbox_synthesis_workers",
        "chatterbox_synthesis_concurrency",
    )
    @classmethod
    def _validate_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be greater than 0")
        return value

    @model_validator(mode="after")
    def _validate_variants_list(self) -> Settings:
        for variant in self.active_variants:
            if variant not in VALID_VARIANTS:
                raise ValueError(
                    f"Invalid variant {variant!r}; must be one of {sorted(VALID_VARIANTS)}"
                )
        if self.chatterbox_segment_min_chars > self.chatterbox_segment_max_chars:
            raise ValueError("segment_min_chars must be <= segment_max_chars")
        return self

    @property
    def active_variants(self) -> list[str]:
        """Return the list of variants to serve."""
        if self.chatterbox_variants:
            return [v.strip().lower() for v in self.chatterbox_variants.split(",") if v.strip()]
        return [self.chatterbox_variant]
