from subtitler.core import text as textutil
from subtitler.core.linebreak import ends_on_function_word
from subtitler.core.models import Cue, Violation
from subtitler.core.rules.base import ERROR, INFO, WARNING, Rule, RuleContext

# Provisional: a line counts as an orphan when it has at most this many
# graphemes and is under this fraction of the longest line. Chosen for
# development only, not from a style guide.
ORPHAN_MAX_GRAPHEMES = 10
ORPHAN_RATIO = 0.33


class MaxLineLengthRule(Rule):
    id = "max-line-length"
    severity = ERROR

    def check(self, cue: Cue, context: RuleContext) -> list[Violation]:
        limit = context.profile.max_chars_per_line
        found = []
        for number, line in enumerate(cue.lines, start=1):
            length = textutil.grapheme_length(line)
            if length > limit:
                message = f"Line {number} has {length} characters, maximum is {limit}"
                found.append(self.violation(cue, message))
        return found


class MaxLineCountRule(Rule):
    id = "max-line-count"
    severity = ERROR

    def check(self, cue: Cue, context: RuleContext) -> list[Violation]:
        limit = context.profile.max_lines
        if len(cue.lines) > limit:
            return [self.violation(cue, f"{len(cue.lines)} lines, maximum is {limit}")]
        return []


class OrphanLineRule(Rule):
    id = "orphan-line"
    severity = INFO

    def check(self, cue: Cue, context: RuleContext) -> list[Violation]:
        if len(cue.lines) < 2:
            return []
        lengths = [textutil.grapheme_length(line) for line in cue.lines]
        shortest = min(lengths)
        longest = max(lengths)
        if shortest <= ORPHAN_MAX_GRAPHEMES and shortest < ORPHAN_RATIO * longest:
            return [self.violation(cue, f"Short line of {shortest} beside one of {longest}")]
        return []


class BreakQualityRule(Rule):
    id = "break-quality"
    severity = WARNING

    def check(self, cue: Cue, context: RuleContext) -> list[Violation]:
        found = []
        for number, line in enumerate(cue.lines[:-1], start=1):
            if ends_on_function_word(line):
                word = line.split()[-1]
                message = f'Line {number} ends on "{word}", which binds to the next line'
                found.append(self.violation(cue, message))
        return found
