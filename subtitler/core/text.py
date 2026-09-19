"""Text measurement for subtitle text.

Every character count and reading-speed figure in the application goes
through this module. Counting is done in extended grapheme clusters, which
is what a reader perceives as a character. For Indic scripts this differs
from the codepoint count because vowel signs and viramas combine with the
preceding consonant.
"""

import unicodedata
from functools import lru_cache

import regex
from uniseg.linebreak import line_break_boundaries

_GRAPHEME = regex.compile(r"\X")
_CHUNK = regex.compile(r"\S+\s*")

# A space followed by one of these is not a break point. UAX #14 already
# agrees for closing brackets and "!?,.;:"; for "%" it would allow the break,
# but "25 %" split across two lines reads badly, so it is kept together.
NO_BREAK_BEFORE = ")]}»”’!?,.;:%"

_SCRIPT_RANGES = [
    ("latin", 0x0041, 0x005A),
    ("latin", 0x0061, 0x007A),
    ("latin", 0x00C0, 0x024F),
    ("latin", 0x1E00, 0x1EFF),
    ("devanagari", 0x0900, 0x097F),
    ("devanagari", 0xA8E0, 0xA8FF),
    ("tamil", 0x0B80, 0x0BFF),
    ("tamil", 0x11FC0, 0x11FFF),
]


def grapheme_length(text: str) -> int:
    # In ASCII every codepoint is its own cluster except CR LF, so skip the regex.
    if text.isascii() and "\r\n" not in text:
        return len(text)
    return len(_GRAPHEME.findall(text))


@lru_cache(maxsize=65536)
def _chunk_breaks(chunk: str) -> tuple[int, ...]:
    return tuple(line_break_boundaries(chunk))[:-1]


def break_opportunities(text: str) -> list[int]:
    """Codepoint offsets where a line may be broken. The start and end of the
    text are not included. A break at offset i means the first line is
    text[:i].

    Inside each space-free chunk the Unicode line breaking algorithm
    (UAX #14) decides, which handles hyphens and scripts written without
    spaces. After a run of spaces a break is allowed unless the next chunk
    starts with closing punctuation. Running UAX #14 over whole strings is
    far too slow for the segmenter, which asks about every candidate cue, so
    results are cached per chunk instead."""
    result = []
    for match in _CHUNK.finditer(text):
        chunk = match.group().rstrip()
        for offset in _chunk_breaks(chunk):
            result.append(match.start() + offset)
        if match.end() < len(text) and text[match.end()] not in NO_BREAK_BEFORE:
            result.append(match.end())
    return result


def _script_of(char: str) -> str:
    code = ord(char)
    for name, low, high in _SCRIPT_RANGES:
        if low <= code <= high:
            return name
    return "other"


def detect_script(text: str) -> str:
    """Return "latin", "tamil", "devanagari", "mixed" or "other".

    Only letters are considered, so punctuation, digits, spaces and
    combining marks never influence the result."""
    found = set()
    for char in text:
        if unicodedata.category(char).startswith("L"):
            found.add(_script_of(char))
    if len(found) > 1:
        return "mixed"
    if len(found) == 1:
        return found.pop()
    return "other"


def chars_per_second(text: str, duration: float) -> float:
    count = grapheme_length(text)
    if count == 0:
        return 0.0
    if duration <= 0:
        return float("inf")
    return count / duration
