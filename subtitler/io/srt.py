"""Reading and writing SubRip (.srt) files.

The reader is lenient because real SRT files often are not: it accepts a
byte order mark, CRLF line endings, a missing final blank line, a full stop
instead of a comma before the milliseconds, missing or wrong cue numbers,
and skips blocks it cannot understand instead of failing.
"""

import re

from subtitler.core.models import Cue

_TIMING = re.compile(
    r"(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})"
)
_BLANK_LINES = re.compile(r"\n\s*\n")


def format_timestamp(seconds: float, separator: str = ",") -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _seconds(hours: str, minutes: str, secs: str, millis: str) -> float:
    # "5" after the separator means 500 ms, so pad on the right.
    return int(hours) * 3600 + int(minutes) * 60 + int(secs) + int(millis.ljust(3, "0")) / 1000


def parse_srt(content: str) -> list[Cue]:
    content = content.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    cues = []
    for block in _BLANK_LINES.split(content.strip()):
        lines = block.split("\n")
        # The timing is normally on line two, but the number line may be missing.
        position = 0
        match = None
        while position < len(lines) and match is None:
            match = _TIMING.search(lines[position])
            position += 1
        if match is None:
            continue
        parts = match.groups()
        text_lines = [line.strip() for line in lines[position:] if line.strip()]
        start = _seconds(*parts[:4])
        end = _seconds(*parts[4:])
        cues.append(Cue(len(cues) + 1, start, end, text_lines))
    return cues


def serialise_srt(cues: list[Cue]) -> str:
    blocks = []
    for number, cue in enumerate(cues, start=1):
        timing = f"{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}"
        blocks.append("\n".join([str(number), timing, *cue.lines]) + "\n")
    return "\n".join(blocks)


def read_srt(path) -> list[Cue]:
    with open(path, encoding="utf-8-sig", errors="replace") as handle:
        return parse_srt(handle.read())


def write_srt(path, cues: list[Cue]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(serialise_srt(cues))
