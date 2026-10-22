import pytest

from subtitler.core.models import Cue
from subtitler.core.rules.profiles import ENGLISH
from subtitler.core.text import grapheme_length
from subtitler.eval.baseline import greedy_segment
from subtitler.eval.metrics import boundary_scores, clean, compliance
from tests.test_segmentation import make_words


def cue(start, end, text="Some ordinary text"):
    return Cue(0, start, end, [text])


def test_boundary_scores_perfect_and_partial():
    reference = [cue(0.0, 2.0), cue(3.0, 5.0), cue(6.0, 8.0)]
    perfect = boundary_scores(reference, reference)
    assert perfect["f1"] == 1.0
    assert perfect["start_error"] == 0.0
    hypothesis = [cue(0.1, 2.0), cue(4.0, 5.0)]
    partial = boundary_scores(hypothesis, reference)
    assert partial["matched"] == 1
    assert partial["precision"] == 0.5
    assert partial["recall"] == pytest.approx(1 / 3)
    assert partial["start_error"] == pytest.approx(0.1)


def test_each_hypothesis_start_matches_once():
    reference = [cue(1.0, 2.0), cue(1.1, 2.0)]
    scores = boundary_scores([cue(1.05, 2.0)], reference)
    assert scores["matched"] == 1


def test_compliance_counts_rule_rates():
    cues = [cue(0.0, 2.0), cue(3.0, 3.2, "Far too much text for so short a time")]
    result = compliance(cues, ENGLISH, [])
    assert result["cues"] == 2
    assert result["rate_reading-speed"] == 0.5
    assert result["error_rate"] == 0.5


def test_markup_is_removed_before_measuring():
    cleaned = clean([Cue(1, 0.0, 1.0, ["<i>Hello</i>", "{\\an8}there"])])
    assert cleaned[0].lines == ["Hello", "there"]


def test_greedy_baseline_respects_limits():
    words = make_words(
        "This is a fairly long stretch of speech that keeps going without any full stop "
        "so the greedy method has to cut it somewhere when the lines are full"
    )
    cues = greedy_segment(words, ENGLISH)
    assert len(cues) > 1
    for result in cues:
        assert len(result.lines) <= ENGLISH.max_lines
        assert all(grapheme_length(line) <= ENGLISH.max_chars_per_line for line in result.lines)
    assert " ".join(c.text() for c in cues) == " ".join(w.text for w in words)


def test_greedy_baseline_closes_at_sentence_end():
    words = make_words("A first sentence here. A second sentence follows it.")
    cues = greedy_segment(words, ENGLISH)
    assert [c.text() for c in cues] == ["A first sentence here.", "A second sentence follows it."]
