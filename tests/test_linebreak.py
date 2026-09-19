from dataclasses import replace

from subtitler.core.linebreak import best_split
from subtitler.core.rules.profiles import LATIN_DEFAULT
from subtitler.core.text import grapheme_length


def profile(max_chars):
    return replace(LATIN_DEFAULT, max_chars_per_line=max_chars)


def test_short_text_stays_on_one_line():
    result = best_split("Hello there", profile(42))
    assert result.lines == ["Hello there"]
    assert result.split_index is None
    assert result.penalty == 0.0


def test_does_not_break_after_the():
    # The most balanced split would leave "the" dangling at the end of line one.
    result = best_split("I think we should visit the museum tomorrow", profile(30))
    assert result.lines == ["I think we should visit", "the museum tomorrow"]


def test_too_long_for_two_lines_returns_none():
    text = " ".join(["word"] * 20)
    assert best_split(text, profile(30)) is None


def test_balanced_split_beats_lopsided():
    text = "one two three four five six seven eight nine ten"
    result = best_split(text, profile(40))
    first, second = result.lines
    assert abs(grapheme_length(first) - grapheme_length(second)) <= 6


def test_longer_second_line_preferred():
    result = best_split("aaaa bbb cccc", profile(9))
    assert result.lines == ["aaaa", "bbb cccc"]


def test_avoids_splitting_hyphenated_compound():
    # "It was a well-" / "known story" is better balanced but splits the
    # compound, and "It was a" ends on an article.
    result = best_split("It was a well-known story", profile(18))
    assert result.lines == ["It was", "a well-known story"]


def test_splits_text_without_spaces():
    text = "日本語の文章を二行に分けて表示します"
    result = best_split(text, profile(10))
    assert len(result.lines) == 2
    assert "".join(result.lines) == text
