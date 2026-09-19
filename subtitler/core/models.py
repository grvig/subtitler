"""Plain data types shared by every layer of the application."""

from dataclasses import dataclass, field

from subtitler.core import text as textutil

# Tolerance for comparing times, so float rounding never creates a violation.
TIME_EPSILON = 1e-6


@dataclass
class Word:
    text: str
    start: float
    end: float
    confidence: float = 1.0
    speaker: str | None = None

    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Cue:
    index: int
    start: float
    end: float
    lines: list[str]
    words: list[Word] = field(default_factory=list)
    locked: bool = False

    def duration(self) -> float:
        return self.end - self.start

    def text(self) -> str:
        return " ".join(self.lines)

    def grapheme_length(self) -> int:
        return textutil.grapheme_length(self.text())

    def chars_per_second(self) -> float:
        return textutil.chars_per_second(self.text(), self.duration())


@dataclass
class SplitResult:
    split_index: int | None
    lines: list[str]
    penalty: float


@dataclass
class Violation:
    rule_id: str
    cue_index: int
    severity: str
    message: str
    timecode: float


@dataclass
class Project:
    media_path: str | None = None
    cues: list[Cue] = field(default_factory=list)
    shot_boundaries: list[float] = field(default_factory=list)
    profile: str = "latin_default"
    dirty: bool = False
