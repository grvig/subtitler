"""A greedy segmenter to compare the optimiser against.

It applies the usual rules one after another, the way transcription tools
typically build subtitles: keep adding words to the current cue, and close
it when the next word would break the character or duration limit, or when a
sentence has ended and the cue is long enough. Lines are filled greedily,
first line first. Cues are timed exactly to their words.
"""

from subtitler.core.models import Cue, Word
from subtitler.core.rules.profiles import StyleProfile
from subtitler.core.segmentation import BOUNDARY_SENTENCE_END, boundary_penalty, span_text
from subtitler.core.text import grapheme_length


def _lines(text: str, limit: int) -> list[str]:
    lines = [""]
    for token in text.split():
        candidate = f"{lines[-1]} {token}".strip()
        if lines[-1] and grapheme_length(candidate) > limit:
            lines.append(token)
        else:
            lines[-1] = candidate
    return lines


def greedy_segment(words: list[Word], profile: StyleProfile) -> list[Cue]:
    capacity = profile.max_chars_per_line * profile.max_lines
    cues = []
    first = 0
    for last in range(len(words)):
        at_end = last + 1 == len(words)
        close = at_end
        if not at_end:
            longer = span_text(words, first, last + 2)
            if grapheme_length(longer) > capacity:
                close = True
            if words[last + 1].end - words[first].start > profile.max_duration:
                close = True
        sentence_done = boundary_penalty(words[last].text) == BOUNDARY_SENTENCE_END
        if sentence_done and words[last].end - words[first].start >= profile.min_duration:
            close = True
        if close:
            text = span_text(words, first, last + 1)
            lines = _lines(text, profile.max_chars_per_line)
            span = list(words[first : last + 1])
            cues.append(Cue(len(cues) + 1, words[first].start, words[last].end, lines, span))
            first = last + 1
    return cues
