import math

from subtitler.core import text


def test_tamil_graphemes_differ_from_codepoints():
    # "தமிழ்" is t-a-m-i-zh: five codepoints, three perceived characters
    word = "தமிழ்"
    assert len(word) == 5
    assert text.grapheme_length(word) == 3


def test_devanagari_graphemes():
    # "नमस्ते" reads as न, म, स्ते. Unicode 15.1 keeps the conjunct together.
    word = "नमस्ते"
    assert len(word) == 6
    assert text.grapheme_length(word) == 3


def test_latin_graphemes_match_len():
    assert text.grapheme_length("hello world") == 11


def test_break_opportunities_without_spaces():
    sample = "日本語の文章です"
    breaks = text.break_opportunities(sample)
    assert breaks
    assert 0 not in breaks
    assert all(0 < b < len(sample) for b in breaks)


def test_break_opportunities_after_spaces():
    assert text.break_opportunities("one two three") == [4, 8]


def test_break_opportunities_match_full_unicode_algorithm():
    from uniseg.linebreak import line_break_boundaries

    samples = [
        "Good evening, and welcome to the programme.",
        "It was a well-known story (really), wasn't it?",
        "நான் வீட்டுக்கு போனேன், சரியா?",
        "The price rose 25 % to $3.50 - a record.",
        "日本語の文章です and some English",
    ]
    for sample in samples:
        expected = list(line_break_boundaries(sample))[:-1]
        # Deliberate difference: never break between a number and " %".
        expected = [b for b in expected if sample[b] != "%"]
        assert text.break_opportunities(sample) == expected, sample


def test_no_break_before_closing_punctuation():
    assert text.break_opportunities("wait ( see ) now") == [5, 7, 13]


def test_detect_script():
    assert text.detect_script("Hello, world!") == "latin"
    assert text.detect_script("நான் வீட்டுக்கு போனேன்") == "tamil"
    assert text.detect_script("नमस्ते दुनिया") == "devanagari"
    assert text.detect_script("I said வணக்கம் to him") == "mixed"


def test_detect_script_ignores_digits_and_punctuation():
    assert text.detect_script("வணக்கம் 2024!") == "tamil"
    assert text.detect_script("123 ... ?") == "other"


def test_chars_per_second():
    assert text.chars_per_second("abcd", 2.0) == 2.0
    assert text.chars_per_second("", 0.0) == 0.0
    assert math.isinf(text.chars_per_second("abc", 0.0))
    assert text.chars_per_second("தமிழ்", 1.5) == 2.0
