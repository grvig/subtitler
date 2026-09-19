"""Style profiles: the limits a subtitle file is checked and segmented against.

PROVISIONAL NUMBERS. Every threshold in this module is a placeholder picked
to make development and testing possible. None of them is taken from a
published style guide. They must be replaced with values from real
broadcaster or industry guidelines before the output is called compliant.
"""

from dataclasses import dataclass, field


def default_cost_weights() -> dict[str, float]:
    # Provisional: relative weights for the segmentation cost terms, not tuned.
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
    max_lines: int
    min_duration: float
    max_duration: float
    min_gap: float
    max_words_per_cue: int = 20
    comfortable_duration: float = 1.5
    cost_weights: dict[str, float] = field(default_factory=default_cost_weights)


# Provisional placeholder values, not sourced from any style guide.
LATIN_DEFAULT = StyleProfile(
    name="latin_default",
    script="latin",
    target_cps=15.0,
    max_cps=20.0,
    max_chars_per_line=42,
    max_lines=2,
    min_duration=1.0,
    max_duration=7.0,
    min_gap=0.08,
    comfortable_duration=1.5,
)

# Provisional placeholder values, not sourced from any style guide. Counts are
# in grapheme clusters, so they are lower than the Latin ones on purpose.
TAMIL = StyleProfile(
    name="tamil",
    script="tamil",
    target_cps=12.0,
    max_cps=17.0,
    max_chars_per_line=36,
    max_lines=2,
    min_duration=1.0,
    max_duration=7.0,
    min_gap=0.08,
    comfortable_duration=1.8,
)

# Very loose limits for tests and debugging. Not meant for real output.
PERMISSIVE = StyleProfile(
    name="permissive",
    script="other",
    target_cps=30.0,
    max_cps=60.0,
    max_chars_per_line=80,
    max_lines=2,
    min_duration=0.1,
    max_duration=20.0,
    min_gap=0.0,
    max_words_per_cue=40,
    comfortable_duration=0.5,
)

PROFILES = {
    LATIN_DEFAULT.name: LATIN_DEFAULT,
    TAMIL.name: TAMIL,
    PERMISSIVE.name: PERMISSIVE,
}
