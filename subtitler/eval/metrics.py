"""Measuring subtitle quality.

Two kinds of measurement, neither of which compares text, because the
recogniser's wording never matches a reference file exactly:

compliance()      how often cues break each rule of the style profile.
                  Needs no reference.
boundary_scores() how well cue start times agree with a reference file:
                  precision, recall and F1 of matching starts within a
                  tolerance, and the mean start and end error of matches.
"""

import math
import re

from subtitler.core.models import Cue
from subtitler.core.rules import RuleEngine
from subtitler.core.rules.base import ERROR
from subtitler.core.rules.profiles import StyleProfile

# Two cue starts within this many seconds count as the same boundary. Chosen
# for this project: about the length of a short word.
BOUNDARY_TOLERANCE = 0.3

_TAG = re.compile(r"<[^>]+>|\{[^}]+\}")


def clean(cues: list[Cue]) -> list[Cue]:
    """Copies of the cues with markup such as <i> removed from the text."""
    cleaned = []
    for cue in cues:
        lines = [_TAG.sub("", line).strip() for line in cue.lines]
        cleaned.append(Cue(cue.index, cue.start, cue.end, [ln for ln in lines if ln], cue.words))
    return cleaned


def compliance(cues: list[Cue], profile: StyleProfile, shots: list[float]) -> dict:
    cues = clean(cues)
    violations = RuleEngine().validate(cues, profile, shots)
    count = len(cues)
    result = {"cues": count}
    if count == 0:
        return result
    flagged = {}
    for violation in violations:
        flagged.setdefault(violation.rule_id, set()).add(violation.cue_index)
    for rule_id in sorted(flagged):
        result[f"rate_{rule_id}"] = len(flagged[rule_id]) / count
    error_cues = {v.cue_index for v in violations if v.severity == ERROR}
    result["error_rate"] = len(error_cues) / count
    # A cue with no duration has infinite CPS; it is already counted as an
    # error above, so it is left out of the mean.
    speeds = [cue.chars_per_second() for cue in cues]
    finite = [speed for speed in speeds if math.isfinite(speed)]
    if finite:
        result["mean_cps"] = sum(finite) / len(finite)
    result["mean_duration"] = sum(cue.duration() for cue in cues) / count
    return result


def boundary_scores(
    hypothesis: list[Cue], reference: list[Cue], tolerance: float = BOUNDARY_TOLERANCE
) -> dict:
    """Match each reference cue start to the nearest unused hypothesis start
    within tolerance, in time order."""
    starts = sorted(cue.start for cue in hypothesis)
    used = [False] * len(starts)
    matches = []
    ordered = sorted(hypothesis, key=lambda cue: cue.start)
    for ref in sorted(reference, key=lambda cue: cue.start):
        best = None
        for position, start in enumerate(starts):
            distance = abs(start - ref.start)
            if used[position] or distance > tolerance:
                continue
            if best is None or distance < abs(starts[best] - ref.start):
                best = position
        if best is not None:
            used[best] = True
            matches.append((ordered[best], ref))
    precision = 0.0
    if hypothesis:
        precision = len(matches) / len(hypothesis)
    recall = 0.0
    if reference:
        recall = len(matches) / len(reference)
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    result = {"precision": precision, "recall": recall, "f1": f1, "matched": len(matches)}
    if matches:
        result["start_error"] = sum(abs(h.start - r.start) for h, r in matches) / len(matches)
        result["end_error"] = sum(abs(h.end - r.end) for h, r in matches) / len(matches)
    return result
