"""Optimal segmentation of a timed word sequence into subtitle cues.

The words are partitioned into consecutive spans, one span per cue, by
dynamic programming:

    dp[0] = 0
    dp[j] = min over i in [max(0, j - K), j - 1] of ( dp[i] + cost(i, j) )
    back[j] = the i that gives the minimum

where cost(i, j) is the cost of a cue holding words[i:j] and K is the
profile's max_words_per_cue. Following back[] from n to 0 recovers the
partition. With K fixed the run time is linear in the number of words.

A cue starts with its first word. It stays on screen for profile.linger
after its last word, or longer if needed to reach min_duration, but always
ends at least min_gap before the next word (see cue_end). "Duration" below
means this display duration.

cost(i, j) is infinite when a hard constraint fails:
  - the cue lasts less than min_duration or more than max_duration
  - best_split() cannot lay the span text out within the line limits
The one exception is a span of a single word. It gets FORCED_CUE_COST
instead, so that an isolated word can never make the whole problem
infeasible. The rule engine still reports the resulting cue.

Otherwise cost(i, j) is the weighted sum of these terms, each weight read
from profile.cost_weights under the same name:
  cps_excess        (cps - target_cps)^2 when cps is above target, else 0
  break_penalty     the penalty of the best line split from best_split()
  boundary_penalty  how bad it is to end a cue after the final word: zero
                    after sentence-final punctuation, small after clause
                    punctuation, large mid-phrase, very large after a
                    function word
  shot_crossings    number of shot boundaries strictly inside the span
  short_penalty     comfortable_duration - duration when positive, else 0
  sentence_crossings  number of sentence ends inside the span, excluding
                    the final word
  internal_pause    total seconds of silence between words inside the
                    span, counting only the part of each gap above
                    PAUSE_THRESHOLD

The weights live on the profile so a sweep script can try different values
with dataclasses.replace(profile, cost_weights={...}).

After the partition is chosen, snap_to_shots moves cue edges onto nearby
shot changes as the Netflix timing guide describes.

Cues marked locked in existing_cues are kept exactly as they are. Their
start and end force cue boundaries, and only the unlocked runs of words
between them are segmented.
"""

import math
from dataclasses import replace

from subtitler.core.linebreak import best_split, is_function_word
from subtitler.core.models import TIME_EPSILON, Cue, Word
from subtitler.core.rules.profiles import StyleProfile
from subtitler.core.text import chars_per_second

# Provisional values, picked by hand during development and not tuned.
BOUNDARY_SENTENCE_END = 0.0
BOUNDARY_CLAUSE = 2.0
BOUNDARY_MID_PHRASE = 10.0
BOUNDARY_FUNCTION_WORD = 40.0
FORCED_CUE_COST = 1000.0
# Silence between two words inside a cue longer than this counts as a pause.
PAUSE_THRESHOLD = 0.3

SENTENCE_END = ".!?…।॥。！？"
CLAUSE_END = ",;:—–、，；："
CLOSING_QUOTES = "\"'”’»)]"


def span_text(words: list[Word], i: int, j: int) -> str:
    return " ".join(word.text.strip() for word in words[i:j])


def boundary_penalty(word: str) -> float:
    stripped = word.strip().rstrip(CLOSING_QUOTES)
    if stripped and stripped[-1] in SENTENCE_END:
        return BOUNDARY_SENTENCE_END
    if stripped and stripped[-1] in CLAUSE_END:
        return BOUNDARY_CLAUSE
    if is_function_word(stripped):
        return BOUNDARY_FUNCTION_WORD
    return BOUNDARY_MID_PHRASE


def cue_end(
    words: list[Word], i: int, j: int, profile: StyleProfile, next_start: float = math.inf
) -> float:
    """Display end time for a cue holding words[i:j].

    The cue stays up for profile.linger after its last word, and longer if
    that is needed to reach min_duration. It always stops min_gap before the
    next word, which is words[j] or, at the end of the list, next_start. When
    the next word comes within the linger time, this closes the gap to exactly
    min_gap, which the Netflix timing guide calls chaining."""
    speech_end = words[j - 1].end
    limit = next_start
    if j < len(words):
        limit = words[j].start
    end = max(speech_end + profile.linger, words[i].start + profile.min_duration)
    end = min(end, limit - profile.min_gap)
    if end <= words[i].start:
        return speech_end
    return end


def cost_terms(
    words: list[Word],
    i: int,
    j: int,
    shot_boundaries: list[float],
    profile: StyleProfile,
    next_start: float = math.inf,
) -> dict[str, float] | None:
    """The unweighted cost terms for a cue holding words[i:j], or None if the
    span breaks a hard constraint."""
    start = words[i].start
    end = cue_end(words, i, j, profile, next_start)
    duration = end - start
    if duration < profile.min_duration - TIME_EPSILON:
        return None
    if duration > profile.max_duration + TIME_EPSILON:
        return None
    text = span_text(words, i, j)
    split = best_split(text, profile)
    if split is None:
        return None
    cps = chars_per_second(text, duration)
    # Counted over the speech only: a cue that merely lingers past a cut is
    # trimmed back to it by snap_to_shots, so that is no reason to avoid it.
    crossings = 0
    for boundary in shot_boundaries:
        if start < boundary < words[j - 1].end:
            crossings += 1
    sentence_crossings = 0
    internal_pause = 0.0
    for k in range(i, j - 1):
        if boundary_penalty(words[k].text) == BOUNDARY_SENTENCE_END:
            sentence_crossings += 1
        pause = words[k + 1].start - words[k].end
        internal_pause += max(0.0, pause - PAUSE_THRESHOLD)
    return {
        "cps_excess": max(0.0, cps - profile.target_cps) ** 2,
        "break_penalty": split.penalty,
        "boundary_penalty": boundary_penalty(words[j - 1].text),
        "shot_crossings": float(crossings),
        "short_penalty": max(0.0, profile.comfortable_duration - duration),
        "sentence_crossings": float(sentence_crossings),
        "internal_pause": internal_pause,
    }


def cost(
    words: list[Word],
    i: int,
    j: int,
    shot_boundaries: list[float],
    profile: StyleProfile,
    next_start: float = math.inf,
) -> float:
    terms = cost_terms(words, i, j, shot_boundaries, profile, next_start)
    if terms is None:
        if j - i == 1:
            return FORCED_CUE_COST
        return math.inf
    total = 0.0
    for name, value in terms.items():
        total += profile.cost_weights.get(name, 0.0) * value
    return total


def _segment_run(
    words: list[Word], shot_boundaries: list[float], profile: StyleProfile, next_start: float
) -> list[Cue]:
    n = len(words)
    dp = [math.inf] * (n + 1)
    back = [0] * (n + 1)
    dp[0] = 0.0
    for j in range(1, n + 1):
        for i in range(j - 1, max(0, j - profile.max_words_per_cue) - 1, -1):
            # Moving i left only lengthens the span, so stop once it is too long.
            duration = cue_end(words, i, j, profile, next_start) - words[i].start
            if j - i > 1 and duration > profile.max_duration:
                break
            candidate = dp[i] + cost(words, i, j, shot_boundaries, profile, next_start)
            if candidate < dp[j]:
                dp[j] = candidate
                back[j] = i
    spans = []
    j = n
    while j > 0:
        spans.append((back[j], j))
        j = back[j]
    spans.reverse()
    cues = []
    for i, j in spans:
        text = span_text(words, i, j)
        split = best_split(text, profile)
        lines = [text]
        if split is not None:
            lines = split.lines
        end = cue_end(words, i, j, profile, next_start)
        cues.append(Cue(0, words[i].start, end, lines, list(words[i:j])))
    return cues


def segment(
    words: list[Word],
    shot_boundaries: list[float],
    profile: StyleProfile,
    existing_cues: list[Cue] | None = None,
) -> list[Cue]:
    boundaries = sorted(shot_boundaries or [])
    locked = sorted((c for c in existing_cues or [] if c.locked), key=lambda c: c.start)
    pieces = []
    position = 0
    for locked_cue in locked:
        run_start = position
        while position < len(words) and _midpoint(words[position]) < locked_cue.start:
            position += 1
        run = words[run_start:position]
        pieces.extend(_segment_run(run, boundaries, profile, locked_cue.start))
        # Words covered by the locked cue are dropped; the cue keeps its own.
        while position < len(words) and _midpoint(words[position]) <= locked_cue.end:
            position += 1
        pieces.append(locked_cue)
    pieces.extend(_segment_run(words[position:], boundaries, profile, math.inf))
    snap_to_shots(pieces, boundaries, profile)
    return [replace(cue, index=number) for number, cue in enumerate(pieces, start=1)]


def snap_to_shots(cues: list[Cue], shot_boundaries: list[float], profile: StyleProfile) -> None:
    """Move cue edges onto nearby shot changes, following the Netflix timing
    guide. Locked cues and cues without word timings are left alone.

    Out-times: a cue whose speech ends before a cut but which lingers across
    it is pulled back to min_gap before the cut, and a cue ending less than
    shot_snap before a cut is extended to min_gap before it. In-times: a cue
    whose speech starts less than shot_snap after a cut starts on the cut.
    Nothing is moved if that would break the duration limits."""
    for position, cue in enumerate(cues):
        if cue.locked or not cue.words:
            continue
        following = None
        if position + 1 < len(cues):
            following = cues[position + 1]
        speech_end = cue.words[-1].end
        for cut in shot_boundaries:
            before_cut = cut - profile.min_gap
            new_end = None
            if speech_end <= before_cut < cue.end:
                new_end = before_cut
            elif cue.end < before_cut < cue.end + profile.shot_snap:
                if following is None or following.start >= cut:
                    new_end = before_cut
            if new_end is not None and _duration_ok(cue.start, new_end, profile):
                cue.end = new_end
    for position, cue in enumerate(cues):
        if cue.locked or not cue.words:
            continue
        previous_end = -math.inf
        if position > 0:
            previous_end = cues[position - 1].end
        for cut in shot_boundaries:
            close_after = cut <= cue.start < cut + profile.shot_snap
            room = previous_end <= cut - profile.min_gap + TIME_EPSILON
            if close_after and room and _duration_ok(cut, cue.end, profile):
                cue.start = cut


def _duration_ok(start: float, end: float, profile: StyleProfile) -> bool:
    duration = end - start
    low = profile.min_duration - TIME_EPSILON
    high = profile.max_duration + TIME_EPSILON
    return low <= duration <= high


def _midpoint(word: Word) -> float:
    return (word.start + word.end) / 2
