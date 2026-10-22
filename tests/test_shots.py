import subprocess
import threading

import pytest

from subtitler.core.models import Cue, Word
from subtitler.core.rules.base import RuleContext
from subtitler.core.rules.profiles import ENGLISH
from subtitler.core.rules.timing import ShotCrossingRule
from subtitler.core.segmentation import segment
from subtitler.pipeline.audio import ffmpeg_available
from subtitler.pipeline.shots import detect_shots
from tests.test_segmentation import make_words

PROFILE = ENGLISH

needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg is not installed")


def make_clip(path, colours=("red", "blue"), seconds=2):
    inputs = []
    for colour in colours:
        inputs += ["-f", "lavfi", "-i", f"color=c={colour}:s=160x120:d={seconds}:r=24"]
    count = len(colours)
    streams = "".join(f"[{n}:v]" for n in range(count))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *inputs]
        + ["-filter_complex", f"{streams}concat=n={count}:v=1[out]", "-map", "[out]", str(path)],
        check=True,
    )


@needs_ffmpeg
def test_detects_a_hard_cut(tmp_path):
    clip = tmp_path / "cut.mp4"
    make_clip(clip)
    progress = []
    cuts = detect_shots(clip, progress.append)
    assert len(cuts) == 1
    assert cuts[0] == pytest.approx(2.0, abs=0.05)
    assert progress[-1] == 100.0


@needs_ffmpeg
def test_detection_stops_when_cancelled(tmp_path):
    clip = tmp_path / "cut.mp4"
    make_clip(clip)
    cancel = threading.Event()
    cancel.set()
    assert detect_shots(clip, None, cancel) == []


def test_unreadable_file_gives_no_cuts(tmp_path):
    path = tmp_path / "not_a_video.mp4"
    path.write_bytes(b"nothing here")
    assert detect_shots(path) == []


def test_linger_is_pulled_back_before_a_cut():
    words = make_words("A short sentence here.")
    cut = words[-1].end + 0.2
    cues = segment(words, [cut], PROFILE)
    assert len(cues) == 1
    assert cues[0].end == pytest.approx(cut - PROFILE.min_gap)


def test_in_time_moves_onto_a_cut_just_before_speech():
    words = make_words("It starts just after the cut.", start=5.3)
    cues = segment(words, [5.0], PROFILE)
    assert cues[0].start == 5.0


def test_out_time_extends_to_a_cut_just_after():
    words = make_words("Speech ends a little before the cut.")
    # Cut 0.3 s after where the linger would end.
    cut = words[-1].end + PROFILE.linger + 0.3
    cues = segment(words, [cut], PROFILE)
    assert cues[0].end == pytest.approx(cut - PROFILE.min_gap)


def test_shot_crossing_rule():
    rule = ShotCrossingRule()
    context = RuleContext(profile=PROFILE, shot_boundaries=[2.0])
    spoken_across = Cue(1, 1.0, 3.0, ["text"], [Word("a", 1.0, 1.5), Word("b", 2.5, 2.9)])
    lingers_across = Cue(1, 1.0, 3.0, ["text"], [Word("a", 1.0, 1.5), Word("b", 1.6, 1.9)])
    no_words = Cue(1, 1.0, 3.0, ["text"])
    assert rule.check(spoken_across, context) == []
    assert len(rule.check(lingers_across, context)) == 1
    assert len(rule.check(no_words, context)) == 1
