"""Writing WebVTT (.vtt) files."""

from subtitler.core.models import Cue
from subtitler.io.srt import format_timestamp


def serialise_vtt(cues: list[Cue]) -> str:
    blocks = ["WEBVTT\n"]
    for number, cue in enumerate(cues, start=1):
        start = format_timestamp(cue.start, ".")
        end = format_timestamp(cue.end, ".")
        blocks.append("\n".join([str(number), f"{start} --> {end}", *cue.lines]) + "\n")
    return "\n".join(blocks)


def write_vtt(path, cues: list[Cue]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(serialise_vtt(cues))
