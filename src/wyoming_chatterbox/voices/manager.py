"""Reference voice file manager with path-traversal protection."""

from __future__ import annotations

import os
from pathlib import Path


class VoiceManager:
    """Manage a directory of reference voice WAV files.

    Only ``.wav`` files directly inside *voices_dir* are exposed; subdirectories
    and other file types are ignored.  Any request that would resolve outside
    *voices_dir* raises ``ValueError``.
    """

    def __init__(self, voices_dir: str | Path) -> None:
        self._dir = Path(voices_dir)

    # -- public API -------------------------------------------------------

    def list_voices(self) -> list[str]:
        """Return sorted list of voice names (file stems, without extension)."""
        if not self._dir.is_dir():
            return []
        return sorted(p.stem for p in self._dir.iterdir() if p.suffix == ".wav" and p.is_file())

    def has_voice(self, name: str) -> bool:
        """Return True if a voice named *name* exists."""
        try:
            self.get_voice_path(name)
            return True
        except (ValueError, FileNotFoundError):
            return False

    def get_voice_path(self, name: str) -> Path:
        """Return the absolute path to the WAV file for *name*.

        Raises:
            ValueError: if *name* is empty, contains path separators, or
                would resolve outside the voices directory.
            FileNotFoundError: if no matching WAV file is found.
        """
        if not name:
            raise ValueError("Voice name must not be empty")

        # Reject anything that looks like a path traversal attempt.
        if os.sep in name or "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(f"Invalid voice name: {name!r}")

        candidate = (self._dir / (name + ".wav")).resolve()
        # Ensure the resolved path is still inside the voices directory.
        try:
            candidate.relative_to(self._dir.resolve())
        except ValueError:
            raise ValueError(f"Voice name {name!r} escapes the voices directory") from None

        if not candidate.is_file():
            raise FileNotFoundError(f"Voice file not found: {candidate}")

        return candidate
