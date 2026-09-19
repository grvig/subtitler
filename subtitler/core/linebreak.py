"""Choosing where to break a cue's text into lines.

Every legal break opportunity (Unicode line breaking, not spaces) is scored,
and so is keeping the text on one line. The cheapest feasible layout wins.
"""

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING

from subtitler.core.models import SplitResult
from subtitler.core.text import break_opportunities, grapheme_length

if TYPE_CHECKING:
    # Imported for annotations only: the rules package imports this module.
    from subtitler.core.rules.profiles import StyleProfile

# Provisional penalty sizes, set by hand during development and not tuned.
# The syntax penalty is kept larger than the worst imbalance penalty so that
# a lopsided but clean break always beats a balanced one after "the".
IMBALANCE_SCALE = 10.0
SECOND_LINE_LONGER_FACTOR = 0.5
SYNTAX_PENALTY = 20.0

OPENING_PUNCTUATION = "([{\"'“‘«¿¡"
HYPHENS = "-‐‑"

# Words a line should not end on, because they bind to what follows.
# One list per script, so supporting a new language means adding data here.
# The Tamil and Devanagari lists are a starting set and need review by a
# native speaker.
FUNCTION_WORDS = {
    "latin": (
        "a an the and but or nor so yet if that because of to in on at by for with "
        "from into onto about over under between through after before than as per "
        "via my your his her its our their this these those"
    ).split(),
    "tamil": "மற்றும் ஆனால் அல்லது என்று என ஒரு அந்த இந்த".split(),
    "devanagari": "और लेकिन या कि तो अगर एक यह वह पर".split(),
}

_FUNCTION_WORD_SET = frozenset(word for words in FUNCTION_WORDS.values() for word in words)


def _strip_punctuation(word: str) -> str:
    kept = [c for c in word if not unicodedata.category(c).startswith("P")]
    return "".join(kept)


def is_function_word(word: str) -> bool:
    return _strip_punctuation(word).lower() in _FUNCTION_WORD_SET


def ends_on_function_word(line: str) -> bool:
    words = line.split()
    if not words:
        return False
    last = words[-1]
    # A trailing comma or full stop means the phrase is closed, so it is fine.
    if unicodedata.category(last[-1]).startswith("P"):
        return False
    return is_function_word(last)


def _imbalance_penalty(first_length: int, second_length: int, limit: int) -> float:
    penalty = IMBALANCE_SCALE * ((first_length - second_length) / limit) ** 2
    # A longer bottom line is easier to read than a longer top line.
    if second_length > first_length:
        penalty = penalty * SECOND_LINE_LONGER_FACTOR
    return penalty


def _syntax_penalty(first: str, second: str) -> float:
    penalty = 0.0
    if ends_on_function_word(first):
        penalty += SYNTAX_PENALTY
    if first[-1] in OPENING_PUNCTUATION:
        penalty += SYNTAX_PENALTY
    if first[-1] in HYPHENS and second[0].isalnum():
        penalty += SYNTAX_PENALTY
    if first[-1].isdigit() and second[0].isdigit():
        penalty += SYNTAX_PENALTY
    return penalty


def best_split(text: str, profile: StyleProfile) -> SplitResult | None:
    """Return the cheapest one or two line layout of text, or None if the
    text cannot be laid out within the line length limit."""
    text = " ".join(text.split())
    limit = profile.max_chars_per_line
    best = None
    total = grapheme_length(text)
    # Each extra line absorbs one space, so anything longer cannot fit.
    if total > limit * profile.max_lines + profile.max_lines - 1:
        return None
    if total <= limit:
        best = SplitResult(split_index=None, lines=[text], penalty=0.0)
    if profile.max_lines < 2:
        return best
    for position in break_opportunities(text):
        first = text[:position].rstrip()
        second = text[position:].lstrip()
        if not first or not second:
            continue
        first_length = grapheme_length(first)
        second_length = grapheme_length(second)
        if first_length > limit or second_length > limit:
            continue
        penalty = _imbalance_penalty(first_length, second_length, limit)
        penalty += _syntax_penalty(first, second)
        if best is None or penalty < best.penalty:
            best = SplitResult(split_index=position, lines=[first, second], penalty=penalty)
    return best
