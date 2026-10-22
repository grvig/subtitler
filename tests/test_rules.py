import pytest

from subtitler.core.models import Cue
from subtitler.core.rules import RuleEngine
from subtitler.core.rules.base import RuleContext
from subtitler.core.rules.layout import (
    BreakQualityRule,
    MaxLineCountRule,
    MaxLineLengthRule,
    OrphanLineRule,
)
from subtitler.core.rules.profiles import ENGLISH, HINDI, TAMIL
from subtitler.core.rules.timing import (
    MaxDurationRule,
    MinDurationRule,
    MinGapRule,
    OverlapRule,
    ReadingSpeedRule,
    ShortGapRule,
)

PROFILE = ENGLISH


def cue(start, end, *lines, index=1):
    return Cue(index=index, start=start, end=end, lines=list(lines))


def context(next_cue=None):
    return RuleContext(profile=PROFILE, next=next_cue)


GOOD = cue(0.0, 2.0, "A perfectly ordinary line")

CASES = [
    (MinDurationRule, GOOD, cue(0.0, 0.3, "Hi"), None),
    (MaxDurationRule, GOOD, cue(0.0, 30.0, "Too long on screen"), None),
    (MinGapRule, GOOD, GOOD, cue(2.01, 3.0, "Next")),
    (ShortGapRule, GOOD, GOOD, cue(2.25, 3.0, "Next")),
    (ReadingSpeedRule, GOOD, cue(0.0, 1.0, "Far too many characters for one second"), None),
    (OverlapRule, GOOD, GOOD, cue(1.5, 3.0, "Next")),
    (MaxLineLengthRule, GOOD, cue(0.0, 5.0, "x" * 60), None),
    (MaxLineCountRule, GOOD, cue(0.0, 5.0, "one", "two", "three"), None),
    (
        OrphanLineRule,
        cue(0.0, 3.0, "The weather is lovely", "and so is the view"),
        cue(0.0, 3.0, "The weather today is really quite lovely", "yes"),
        None,
    ),
    (
        BreakQualityRule,
        cue(0.0, 3.0, "I walked to the shop,", "and bought some bread"),
        cue(0.0, 3.0, "I walked to the", "shop and bought bread"),
        None,
    ),
]


@pytest.mark.parametrize("rule_class, passing, failing, next_cue", CASES)
def test_rule_pass_and_fail(rule_class, passing, failing, next_cue):
    rule = rule_class()
    # For the neighbour rules the passing case has a well spaced next cue.
    far_next = cue(10.0, 12.0, "Later")
    if next_cue is None:
        assert rule.check(passing, context()) == []
    else:
        assert rule.check(passing, context(far_next)) == []
    violations = rule.check(failing, context(next_cue))
    assert len(violations) == 1
    assert violations[0].rule_id == rule.id
    assert violations[0].cue_index == failing.index


def test_engine_reports_neighbour_violations():
    cues = [
        cue(0.0, 2.0, "First cue here", index=1),
        cue(1.5, 3.5, "Second cue here", index=2),
    ]
    violations = RuleEngine().validate(cues, PROFILE, [])
    assert [v.rule_id for v in violations] == ["overlap"]
    assert violations[0].cue_index == 1
    assert violations[0].timecode == 0.0


def test_engine_clean_file():
    cues = [
        cue(0.0, 2.0, "First cue here", index=1),
        cue(2.5, 4.5, "Second cue here", index=2),
    ]
    assert RuleEngine().validate(cues, PROFILE) == []


def test_profiles_carry_netflix_limits():
    for profile in [ENGLISH, HINDI, TAMIL]:
        assert profile.max_chars_per_line == 42
        assert profile.max_lines == 2
        assert abs(profile.min_duration - 5 / 6) < 1e-9
        assert profile.max_duration == 7.0
        assert abs(profile.min_gap - 2 / 24) < 1e-9
    assert ENGLISH.max_cps == 20.0
    assert HINDI.max_cps == TAMIL.max_cps == 22.0
    assert HINDI.script == "devanagari"
