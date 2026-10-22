"""Giving corrected cue text its own word timings.

When someone fixes a cue's text, the word timings that came from the
recogniser no longer match it. The audio of that cue is transcribed again
on its own (see pipeline/jobs.py), and align_words() lines the corrected
words up against what the recogniser heard: words that match take the
recognised timing, and the rest are spread over the gaps between them.
"""

from difflib import SequenceMatcher

import regex

from subtitler.core.models import Word
from subtitler.core.text import grapheme_length

_PUNCTUATION = regex.compile(r"[\p{P}\p{S}]+")


def _key(token: str) -> str:
    return _PUNCTUATION.sub("", token).casefold()


def needs_alignment(cue) -> bool:
    """True for a cue with no word timings, or whose text no longer matches
    the words it holds."""
    if not cue.words:
        return True
    spoken = " ".join(word.text for word in cue.words)
    return spoken.split() != cue.text().split()


def align_words(text: str, recognised: list[Word], start: float, end: float) -> list[Word]:
    """Timed words for text, within [start, end], using recognised words as
    anchors wherever they match."""
    tokens = text.split()
    if not tokens:
        return []
    timings = [None] * len(tokens)
    matcher = SequenceMatcher(
        None, [_key(t) for t in tokens], [_key(w.text) for w in recognised], autojunk=False
    )
    for tag, first, first_end, second, second_end in matcher.get_opcodes():
        # Equal runs match word for word; a replaced run of the same length is
        # almost always the same words heard slightly differently.
        if tag == "equal" or (tag == "replace" and first_end - first == second_end - second):
            for offset in range(first_end - first):
                timings[first + offset] = recognised[second + offset]
    words = []
    position = 0
    while position < len(tokens):
        anchor = timings[position]
        if anchor is not None:
            words.append(Word(tokens[position], anchor.start, anchor.end, anchor.confidence))
            position += 1
            continue
        # A run of unmatched tokens fills the time between its neighbours,
        # shared out by length.
        run_end = position
        while run_end < len(tokens) and timings[run_end] is None:
            run_end += 1
        left = start
        if words:
            left = words[-1].end
        right = end
        if run_end < len(tokens):
            right = timings[run_end].start
        right = max(right, left)
        lengths = [max(1, grapheme_length(t)) for t in tokens[position:run_end]]
        total = sum(lengths)
        clock = left
        for token, length in zip(tokens[position:run_end], lengths, strict=True):
            step = (right - left) * length / total
            words.append(Word(token, clock, clock + step, 0.0))
            clock += step
        position = run_end
    return words
