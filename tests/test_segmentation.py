"""Tests for the text segmenter."""

from __future__ import annotations

from wyoming_chatterbox.segmentation.segmenter import TextSegmenter


def make_segmenter(**kwargs) -> TextSegmenter:
    defaults = {"min_chars": 10, "target_chars": 40, "max_chars": 80}
    defaults.update(kwargs)
    return TextSegmenter(**defaults)


def test_short_sentence_no_split():
    seg = make_segmenter()
    assert seg.feed("Hello there") == []
    assert seg.flush() == ["Hello there"]


def test_paragraph_split():
    seg = make_segmenter(min_chars=1)
    out = seg.feed("First paragraph.\n\nSecond paragraph text here.")
    assert "First paragraph." in out[0]


def test_sentence_split_period():
    seg = make_segmenter()
    out = seg.feed("This is the first sentence. This is the second one now. ")
    assert out[0] == "This is the first sentence."


def test_sentence_split_exclaim_question():
    seg = make_segmenter()
    out = seg.feed("Are you sure about this? Yes I am totally sure! ")
    assert out[0] == "Are you sure about this?"
    out2 = seg.flush()
    assert any("Yes I am" in s for s in out + out2)


def test_no_split_on_decimal():
    seg = make_segmenter(min_chars=1)
    out = seg.feed("The value of pi is 3.14 in this text. ")
    assert out == ["The value of pi is 3.14 in this text."]


def test_no_split_on_abbreviation():
    seg = make_segmenter(min_chars=1)
    out = seg.feed("We met Dr. Smith at the clinic yesterday. ")
    assert out == ["We met Dr. Smith at the clinic yesterday."]


def test_comma_split_over_target():
    seg = make_segmenter(min_chars=5, target_chars=20, max_chars=200)
    text = "one two three four five, and here comes the rest of the sentence "
    out = seg.feed(text)
    assert out
    assert out[0].endswith(",")


def test_max_chars_hard_split():
    seg = make_segmenter(min_chars=5, target_chars=1000, max_chars=30)
    text = "word " * 20  # no punctuation, long
    out = seg.feed(text)
    assert out
    assert all(len(s) <= 30 for s in out)


def test_flush_returns_remainder():
    seg = make_segmenter()
    seg.feed("Some leftover text without terminator")
    remainder = seg.flush()
    assert remainder == ["Some leftover text without terminator"]


def test_incremental_feed():
    seg = make_segmenter()
    assert seg.feed("This is a sen") == []
    assert seg.feed("tence that is now complete. And ") == [
        "This is a sentence that is now complete."
    ]
    assert seg.flush() == ["And"]


def test_no_split_on_ellipsis():
    seg = make_segmenter(min_chars=1)
    out = seg.feed("Well... I am not so sure about that decision. ")
    assert out[0].startswith("Well...")
    assert out[0].endswith("decision.")
