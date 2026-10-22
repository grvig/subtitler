import os

import numpy as np

from subtitler.core.models import Cue, Project, Word
from subtitler.io import project as project_io
from subtitler.io.srt import format_timestamp, parse_srt, read_srt, serialise_srt, write_srt
from subtitler.io.vtt import serialise_vtt

CANONICAL = (
    "1\n"
    "00:00:01,000 --> 00:00:03,500\n"
    "Good evening,\n"
    "and welcome.\n"
    "\n"
    "2\n"
    "00:00:04,250 --> 01:02:03,004\n"
    "வணக்கம்\n"
)


def test_srt_round_trip():
    assert serialise_srt(parse_srt(CANONICAL)) == CANONICAL


def test_srt_parsed_values():
    cues = parse_srt(CANONICAL)
    assert len(cues) == 2
    assert cues[0].start == 1.0
    assert cues[0].end == 3.5
    assert cues[0].lines == ["Good evening,", "and welcome."]
    assert cues[1].end == 3723.004


def test_srt_malformed_input_does_not_raise():
    messy = (
        "﻿1\r\n"
        "00:00:01.000 --> 00:00:02.5\r\n"
        "Period separators and CRLF\r\n"
        "\r\n"
        "this block has no timing line\r\n"
        "\r\n\r\n\r\n"
        "00:00:05,000 --> 00:00:06,000\r\n"
        "No cue number and no trailing blank line"
    )
    cues = parse_srt(messy)
    assert [cue.lines for cue in cues] == [
        ["Period separators and CRLF"],
        ["No cue number and no trailing blank line"],
    ]
    assert cues[0].end == 2.5
    assert [cue.index for cue in cues] == [1, 2]


def test_srt_garbage_gives_empty_list():
    assert parse_srt("") == []
    assert parse_srt("not a subtitle file at all") == []


def test_srt_file_round_trip_with_bom(tmp_path):
    path = tmp_path / "in.srt"
    path.write_bytes(("﻿" + CANONICAL).encode("utf-8"))
    cues = read_srt(path)
    out = tmp_path / "out.srt"
    write_srt(out, cues)
    assert out.read_text(encoding="utf-8") == CANONICAL


def test_timestamp_rounding():
    assert format_timestamp(1.9996) == "00:00:02,000"
    assert format_timestamp(0.0, ".") == "00:00:00.000"


def test_vtt_output():
    cues = [Cue(1, 1.0, 2.0, ["Hello", "there"])]
    assert serialise_vtt(cues) == "WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.000\nHello\nthere\n"


def make_project():
    words = [Word("Hello", 0.0, 0.5), Word("there.", 0.6, 1.0, 0.8, "A")]
    cue = Cue(1, 0.0, 1.0, ["Hello there."], words, locked=True)
    return Project("clip.mp4", [cue], [12.5], "tamil")


def test_project_round_trip(tmp_path):
    path = tmp_path / "demo.subproj"
    project = make_project()
    project_io.save_project(project, path)
    loaded = project_io.load_project(path)
    assert loaded == project


def test_recovery_offered_only_when_autosave_is_newer(tmp_path):
    path = tmp_path / "demo.subproj"
    project = make_project()
    project_io.save_project(project, path)
    assert project_io.find_recovery(path) is None
    project_io.write_autosave(project, path)
    autosave = project_io.autosave_path(path)
    os.utime(path, (1000, 1000))
    os.utime(autosave, (2000, 2000))
    assert project_io.find_recovery(path) == autosave
    os.utime(autosave, (500, 500))
    assert project_io.find_recovery(path) is None
    project_io.discard_autosave(path)
    assert not os.path.exists(autosave)


def test_peaks_round_trip(tmp_path):
    path = tmp_path / "demo.subproj"
    assert project_io.load_peaks(path) is None
    mins = np.array([-0.5, -0.25], dtype=np.float32)
    maxs = np.array([0.5, 0.75], dtype=np.float32)
    project_io.save_peaks(path, mins, maxs, 100)
    loaded_min, loaded_max, rate = project_io.load_peaks(path)
    assert loaded_min.tolist() == mins.tolist()
    assert loaded_max.tolist() == maxs.tolist()
    assert rate == 100
