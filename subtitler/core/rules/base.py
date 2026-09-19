from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from subtitler.core.models import Cue, Violation
from subtitler.core.rules.profiles import StyleProfile

ERROR = "error"
WARNING = "warning"
INFO = "info"


@dataclass
class RuleContext:
    profile: StyleProfile
    previous: Cue | None = None
    next: Cue | None = None
    shot_boundaries: list[float] = field(default_factory=list)


class Rule(ABC):
    id = "rule"
    severity = WARNING

    @abstractmethod
    def check(self, cue: Cue, context: RuleContext) -> list[Violation]:
        """Return the violations this rule finds in cue, or an empty list."""

    def violation(self, cue: Cue, message: str) -> Violation:
        return Violation(self.id, cue.index, self.severity, message, cue.start)
