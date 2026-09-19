from subtitler.core.models import Cue, Violation
from subtitler.core.rules.base import RuleContext
from subtitler.core.rules.layout import (
    BreakQualityRule,
    MaxLineCountRule,
    MaxLineLengthRule,
    OrphanLineRule,
)
from subtitler.core.rules.profiles import StyleProfile
from subtitler.core.rules.timing import (
    MaxDurationRule,
    MinDurationRule,
    MinGapRule,
    OverlapRule,
    ReadingSpeedRule,
)

# Adding a rule is one line here.
DEFAULT_RULES = [
    MinDurationRule,
    MaxDurationRule,
    MinGapRule,
    ReadingSpeedRule,
    OverlapRule,
    MaxLineLengthRule,
    MaxLineCountRule,
    OrphanLineRule,
    BreakQualityRule,
]


class RuleEngine:
    def __init__(self, rules=None):
        if rules is None:
            rules = [rule_class() for rule_class in DEFAULT_RULES]
        self.rules = rules

    def validate(
        self,
        cues: list[Cue],
        profile: StyleProfile,
        shot_boundaries: list[float] | None = None,
    ) -> list[Violation]:
        boundaries = list(shot_boundaries or [])
        found = []
        for position, cue in enumerate(cues):
            context = RuleContext(profile=profile, shot_boundaries=boundaries)
            if position > 0:
                context.previous = cues[position - 1]
            if position + 1 < len(cues):
                context.next = cues[position + 1]
            for rule in self.rules:
                found.extend(rule.check(cue, context))
        return found
