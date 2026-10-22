from dataclasses import replace

from subtitler.core.models import Cue, Word
from subtitler.core.rules.profiles import ENGLISH
from subtitler.core.segmentation import segment

PROFILE = ENGLISH


def make_words(sentence, word_length=0.4, gap=0.05, start=0.0):
    words = []
    time = start
    for token in sentence.split():
        words.append(Word(token, time, time + word_length))
        time += word_length + gap
    return words


def test_empty_input():
    assert segment([], [], PROFILE) == []


def test_splits_at_sentence_boundary():
    words = make_words(
        "I have been waiting here for over an hour now. "
        "Nobody told me the train was going to be this late.",
        word_length=0.3,
    )
    cues = segment(words, [], PROFILE)
    assert len(cues) == 2
    assert cues[0].text().endswith("hour now.")
    assert cues[1].text().startswith("Nobody")
    assert [cue.index for cue in cues] == [1, 2]


def test_cue_timing_follows_words():
    words = make_words("Short and simple.")
    cues = segment(words, [], PROFILE)
    assert len(cues) == 1
    assert cues[0].start == words[0].start
    # Nothing follows, so the cue lingers after the last word.
    assert cues[0].end == words[-1].end + PROFILE.linger
    assert cues[0].words == words


def test_locked_cue_survives_resegmentation():
    words = make_words(
        "The first sentence is here. The second one follows it. "
        "And then a third sentence ends the example."
    )
    cues = segment(words, [], PROFILE)
    assert len(cues) >= 2
    edited = replace(cues[0], lines=["Completely rewritten", "by the editor"], locked=True)
    existing = [edited] + cues[1:]
    result = segment(words, [], PROFILE, existing_cues=existing)
    survivors = [cue for cue in result if cue.locked]
    assert len(survivors) == 1
    kept = survivors[0]
    assert kept.lines == edited.lines
    assert kept.start == edited.start
    assert kept.end == edited.end
    assert kept.words == edited.words
    # No other cue reuses the locked cue's words.
    for cue in result:
        if cue is not kept:
            assert cue.end <= kept.start or cue.start >= kept.end


def test_shot_boundary_pulls_split_to_the_cut():
    words = make_words("Tom quickly painted every single wall bright green today")
    assert len(segment(words, [], PROFILE)) == 1
    cut = (words[2].end + words[3].start) / 2
    cues = segment(words, [cut], PROFILE)
    assert len(cues) == 2
    assert cues[0].end <= cut <= cues[1].start
    assert cues[0].text() == "Tom quickly painted"


def test_isolated_short_word_does_not_break_segmentation():
    # The pause is longer than max_duration, so "Yes." cannot join the next words
    # and is too short to be a legal cue by itself.
    words = [Word("Yes.", 0.0, 0.2)] + make_words("That is exactly right.", start=8.0)
    cues = segment(words, [], PROFILE)
    assert cues[0].text() == "Yes."
    assert "".join(cue.text() for cue in cues).replace(" ", "") == "Yes.Thatisexactlyright."


def test_locked_cue_in_the_middle_forces_both_boundaries():
    words = make_words("one two three four five six seven eight nine ten eleven twelve")
    locked = Cue(9, words[4].start, words[7].end, ["LOCKED"], list(words[4:8]), locked=True)
    result = segment(words, [], PROFILE, existing_cues=[locked])
    texts = [cue.text() for cue in result]
    position = texts.index("LOCKED")
    assert result[position].start == locked.start
    assert result[position].end == locked.end
    assert " ".join(texts[:position]) == "one two three four"
    assert " ".join(texts[position + 1 :]) == "nine ten eleven twelve"
    assert [cue.index for cue in result] == list(range(1, len(result) + 1))


def test_pause_and_sentence_end_are_not_bridged():
    # Merging would lower the CPS of the second sentence by absorbing the pause.
    words = make_words("Welcome to the show.", word_length=0.25)
    words += make_words("Tonight we look at the city", word_length=0.25, start=words[-1].end + 0.8)
    cues = segment(words, [], PROFILE)
    assert cues[0].text() == "Welcome to the show."


def test_short_cue_is_held_into_following_silence():
    words = [Word("Yes.", 0.0, 0.3)] + make_words("That is exactly right.", start=3.0)
    cues = segment(words, [], PROFILE)
    assert cues[0].text() == "Yes."
    assert cues[0].end == PROFILE.min_duration


def test_min_gap_is_applied_between_cues():
    words = make_words("The first sentence ends here. The second one starts right away.", gap=0.02)
    cues = segment(words, [], PROFILE)
    assert len(cues) == 2
    assert abs(cues[1].start - cues[0].end - PROFILE.min_gap) < 1e-9


def test_linger_is_cut_short_by_next_word():
    # Next sentence starts 0.3 s after the first ends: inside the linger time,
    # so the gap closes to exactly min_gap instead of leaving a short gap.
    words = make_words("The first sentence ends here.")
    words += make_words("The second one follows soon.", start=words[-1].end + 0.3)
    cues = segment(words, [], PROFILE)
    assert len(cues) == 2
    assert abs(cues[1].start - cues[0].end - PROFILE.min_gap) < 1e-9
