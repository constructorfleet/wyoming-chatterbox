"""Incremental text segmentation for streaming synthesis."""

from __future__ import annotations

import re

# Abbreviations that should not trigger a sentence split when followed by a period.
_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "inc",
    "ltd",
    "co",
    "corp",
    "no",
    "vol",
    "fig",
    "gen",
    "gov",
    "sen",
    "rep",
    "col",
    "capt",
    "sgt",
    "lt",
    "cmdr",
    "adm",
    "rev",
    "hon",
    "u.s",
    "u.k",
    "e.g",
    "i.e",
    "a.m",
    "p.m",
}

_URL_RE = re.compile(r"https?://\S+$", re.IGNORECASE)


class TextSegmenter:
    """Split incrementally-fed text into synthesis-friendly segments."""

    def __init__(
        self,
        min_chars: int = 40,
        target_chars: int = 160,
        max_chars: int = 280,
    ) -> None:
        if min_chars <= 0 or target_chars <= 0 or max_chars <= 0:
            raise ValueError("char limits must be positive")
        if min_chars > max_chars:
            raise ValueError("min_chars must be <= max_chars")
        self.min_chars = min_chars
        self.target_chars = target_chars
        self.max_chars = max_chars
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        """Feed more text and return any complete segments ready for synthesis."""
        self._buffer += text
        segments: list[str] = []
        while True:
            boundary = self._find_boundary(self._buffer)
            if boundary is None:
                break
            segment = self._buffer[:boundary].strip()
            self._buffer = self._buffer[boundary:].lstrip()
            if segment:
                segments.append(segment)
        return segments

    def flush(self) -> list[str]:
        """Return any remaining buffered text as final segment(s)."""
        remainder = self._buffer.strip()
        self._buffer = ""
        if not remainder:
            return []
        # The remainder may still exceed max_chars; hard-split it.
        segments: list[str] = []
        while len(remainder) > self.max_chars:
            split_at = self._hard_split_point(remainder)
            segments.append(remainder[:split_at].strip())
            remainder = remainder[split_at:].lstrip()
        if remainder:
            segments.append(remainder)
        return [s for s in segments if s]

    # -- internal helpers -------------------------------------------------

    def _find_boundary(self, buffer: str) -> int | None:
        """Return an index at which to split the buffer, or None if not ready."""
        # 1. Paragraph break.
        para = buffer.find("\n\n")
        if para != -1:
            return para + 2

        # 2. Sentence-ending punctuation followed by whitespace.
        for match in re.finditer(r"[.!?]+[\"')\]]?(\s)", buffer):
            end = match.end()
            if self._is_valid_sentence_end(buffer, match.start()):
                if end >= self.min_chars:
                    return end

        # 3. Semicolon / colon boundaries.
        for match in re.finditer(r"[;:](\s)", buffer):
            end = match.end()
            if end >= self.min_chars:
                return end

        # 4. Comma boundary once we exceed the target length.
        if len(buffer) > self.target_chars:
            for match in re.finditer(r",(\s)", buffer):
                end = match.end()
                if end >= self.min_chars:
                    return end

        # 5 & 6. Length-based split once the buffer exceeds max_chars.
        if len(buffer) > self.max_chars:
            return self._hard_split_point(buffer)

        return None

    def _hard_split_point(self, buffer: str) -> int:
        """Find a whitespace split near ``max_chars`` or hard-cut at ``max_chars``."""
        window = buffer[: self.max_chars]
        space = window.rfind(" ")
        if space > self.min_chars:
            return space + 1
        return self.max_chars

    def _is_valid_sentence_end(self, buffer: str, punct_index: int) -> bool:
        """Return True if the punctuation at ``punct_index`` ends a real sentence."""
        char = buffer[punct_index]

        # Ellipsis: treat "..." as non-terminal (avoid mid-thought splits).
        if char == "." and buffer[punct_index : punct_index + 3] == "...":
            return False
        if char == "." and punct_index > 0 and buffer[punct_index - 1] == ".":
            return False

        # Decimal numbers: digit before AND after the period.
        if char == "." and 0 < punct_index < len(buffer) - 1:
            if buffer[punct_index - 1].isdigit() and buffer[punct_index + 1].isdigit():
                return False

        # Preceding token being an abbreviation or single initial.
        prefix = buffer[:punct_index]
        token_match = re.search(r"([A-Za-z][A-Za-z.]*)$", prefix)
        if token_match:
            token = token_match.group(1).lower().rstrip(".")
            if token in _ABBREVIATIONS:
                return False
            # Single-letter initial like "J." in "J.K."
            if len(token) == 1 and char == ".":
                return False

        # URLs.
        if _URL_RE.search(prefix):
            return False

        return True
