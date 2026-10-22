"""Style profiles: the limits a subtitle file is checked and segmented against.

Limits marked "Netflix" come from the Netflix Timed Text Style Guides
(partnerhelp.netflixstudios.com), adult programmes, subtitles rather than SDH:

  General Requirements: 2 lines at most, 5/6 s minimum and 7 s maximum
      duration per event.
  Subtitle Timing Guidelines: at least 2 frames between events; gaps should
      be either 2 frames or at least half a second; an out-time should ideally
      sit half a second after the audio ends when no subtitle follows;
      dialogue starting within half a second after a shot change starts on
      it, and out-times near a shot change end two frames before it.
  English (USA): 42 characters per line, up to 20 characters per second.
  Hindi: 42 characters per line, up to 22 characters per second.
  Tamil: 42 characters per line, up to 22 characters per second.

Frame-based values assume 24 fps. The guides do not say how characters are
counted for Indic scripts; this project counts grapheme clusters, which is
the more lenient reading.

Values marked "provisional" are choices made for this project and do not
come from any guide: the target reading speed the segmenter aims for, the
comfortable duration, and the cost weights. They are tuning knobs.
"""

from dataclasses import dataclass, field

FRAME = 1 / 24

# Netflix: general requirements and subtitle timing guidelines.
NETFLIX_MAX_LINES = 2
NETFLIX_MIN_DURATION = 5 / 6
NETFLIX_MAX_DURATION = 7.0
NETFLIX_MIN_GAP = 2 * FRAME
NETFLIX_CHAIN_GAP = 0.5
NETFLIX_LINGER = 0.5
NETFLIX_SHOT_SNAP = 0.5


def default_cost_weights() -> dict[str, float]:
    # Provisional: relative weights for the segmentation cost terms.
    return {
        "cps_excess": 1.0,
        "break_penalty": 1.0,
        "boundary_penalty": 1.0,
        "shot_crossings": 50.0,
        "short_penalty": 5.0,
        "sentence_crossings": 20.0,
        "internal_pause": 20.0,
    }


@dataclass
class StyleProfile:
    name: str
    script: str
    target_cps: float
    max_cps: float
    max_chars_per_line: int
    max_lines: int = NETFLIX_MAX_LINES
    min_duration: float = NETFLIX_MIN_DURATION
    max_duration: float = NETFLIX_MAX_DURATION
    min_gap: float = NETFLIX_MIN_GAP
    # Gaps shorter than this but longer than min_gap should be closed.
    chain_gap: float = NETFLIX_CHAIN_GAP
    # How long a cue may stay up after its last word ends.
    linger: float = NETFLIX_LINGER
    # Cue edges this close to a shot change are moved onto it.
    shot_snap: float = NETFLIX_SHOT_SNAP
    max_words_per_cue: int = 20
    comfortable_duration: float = 1.5
    cost_weights: dict[str, float] = field(default_factory=default_cost_weights)


ENGLISH = StyleProfile(
    name="english",
    script="latin",
    # Provisional target; the Netflix maximum is 20.
    target_cps=15.0,
    max_cps=20.0,
    max_chars_per_line=42,
)

HINDI = StyleProfile(
    name="hindi",
    script="devanagari",
    # Provisional target; the Netflix maximum is 22.
    target_cps=17.0,
    max_cps=22.0,
    max_chars_per_line=42,
    comfortable_duration=1.8,
)

TAMIL = StyleProfile(
    name="tamil",
    script="tamil",
    # Provisional target; the Netflix maximum is 22.
    target_cps=17.0,
    max_cps=22.0,
    max_chars_per_line=42,
    comfortable_duration=1.8,
)

# Very loose limits for tests and debugging. Not meant for real output.
PERMISSIVE = StyleProfile(
    name="permissive",
    script="other",
    target_cps=30.0,
    max_cps=60.0,
    max_chars_per_line=80,
    min_duration=0.1,
    max_duration=20.0,
    min_gap=0.0,
    chain_gap=0.0,
    linger=0.0,
    max_words_per_cue=40,
    comfortable_duration=0.5,
)

PROFILES = {profile.name: profile for profile in [ENGLISH, HINDI, TAMIL, PERMISSIVE]}
DEFAULT_PROFILE = ENGLISH
