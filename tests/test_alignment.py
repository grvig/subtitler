import pytest

from subtitler.core.alignment import align_words, needs_alignment
from subtitler.core.models import Cue, Word


def heard(*items):
    return [Word(text, start, end, 0.9) for text, start, end in items]


def test_matching_words_take_recognised_timing():
    recognised = heard(("Hello,", 1.0, 1.4), ("there", 1.5, 1.9))
    words = align_words("hello there!", recognised, 1.0, 2.0)
    assert [(w.text, w.start, w.end) for w in words] == [("hello", 1.0, 1.4), ("there!", 1.5, 1.9)]


def test_corrected_word_takes_the_place_of_a_misheard_one():
    # "billig Fluggesellschaft" was corrected to one word; "billig" was heard.
    recognised = heard(("eine", 1.0, 1.2), ("billig", 1.3, 1.6), ("Fluggesellschaft", 1.7, 2.4))
    words = align_words("eine Billigfluggesellschaft", recognised, 1.0, 2.5)
    assert [w.text for w in words] == ["eine", "Billigfluggesellschaft"]
    assert words[0].start == 1.0
    assert words[1].start >= words[0].end
    assert words[1].end <= 2.5


def test_inserted_words_are_spread_between_neighbours():
    recognised = heard(("one", 0.0, 0.5), ("four", 2.0, 2.5))
    words = align_words("one two three four", recognised, 0.0, 3.0)
    assert [w.text for w in words] == ["one", "two", "three", "four"]
    assert words[1].start == pytest.approx(0.5)
    assert words[2].end == pytest.approx(2.0)
    assert words[1].confidence == 0.0


def test_nothing_heard_spreads_text_over_the_cue():
    words = align_words("a bb ccc", [], 1.0, 4.0)
    assert words[0].start == 1.0
    assert words[-1].end == pytest.approx(4.0)
    assert all(a.end == pytest.approx(b.start) for a, b in zip(words, words[1:], strict=False))


def test_needs_alignment():
    words = [Word("Hello", 0.0, 0.5), Word("there.", 0.6, 1.0)]
    assert not needs_alignment(Cue(1, 0.0, 1.0, ["Hello", "there."], words))
    assert needs_alignment(Cue(1, 0.0, 1.0, ["Hello", "friend."], words))
    assert needs_alignment(Cue(1, 0.0, 1.0, ["Imported"]))
