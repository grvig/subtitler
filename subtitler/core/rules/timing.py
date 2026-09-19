from subtitler.core.models import TIME_EPSILON, Cue, Violation
from subtitler.core.rules.base import ERROR, WARNING, Rule, RuleContext


class MinDurationRule(Rule):
    id = "min-duration"
    severity = WARNING

    def check(self, cue: Cue, context: RuleContext) -> list[Violation]:
        limit = context.profile.min_duration
        if cue.duration() < limit - TIME_EPSILON:
            return [self.violation(cue, f"On screen {cue.duration():.2f}s, minimum is {limit}s")]
        return []


class MaxDurationRule(Rule):
    id = "max-duration"
    severity = WARNING

    def check(self, cue: Cue, context: RuleContext) -> list[Violation]:
        limit = context.profile.max_duration
        if cue.duration() > limit + TIME_EPSILON:
            return [self.violation(cue, f"On screen {cue.duration():.2f}s, maximum is {limit}s")]
        return []


class MinGapRule(Rule):
    id = "min-gap"
    severity = WARNING

    def check(self, cue: Cue, context: RuleContext) -> list[Violation]:
        if context.next is None:
            return []
        gap = context.next.start - cue.end
        limit = context.profile.min_gap
        # Negative gaps are overlaps and are reported by OverlapRule instead.
        if 0 <= gap < limit - TIME_EPSILON:
            return [self.violation(cue, f"Gap to next cue {gap:.3f}s, minimum is {limit}s")]
        return []


class ReadingSpeedRule(Rule):
    id = "reading-speed"
    severity = ERROR

    def check(self, cue: Cue, context: RuleContext) -> list[Violation]:
        cps = cue.chars_per_second()
        limit = context.profile.max_cps
        if cps > limit:
            return [self.violation(cue, f"Reading speed {cps:.1f} cps, maximum is {limit}")]
        return []


class OverlapRule(Rule):
    id = "overlap"
    severity = ERROR

    def check(self, cue: Cue, context: RuleContext) -> list[Violation]:
        if context.next is None:
            return []
        if context.next.start < cue.end - TIME_EPSILON:
            overlap = cue.end - context.next.start
            return [self.violation(cue, f"Overlaps next cue by {overlap:.3f}s")]
        return []
