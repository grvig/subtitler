import copy

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from subtitler.io.srt import read_srt
from subtitler.pipeline import jobs
from subtitler.ui.main_window import MainWindow
from tests.test_pipeline import write_tone


@pytest.fixture
def window(qtbot, monkeypatch, tmp_path):
    def fake_extract(media_path, out_path):
        write_tone(out_path, seconds=20.0)

    monkeypatch.setattr(jobs, "extract_audio", fake_extract)
    monkeypatch.setattr("subtitler.ui.main_window.ffmpeg_available", lambda: True)
    # Answer every confirmation with Yes instead of blocking on a modal dialog.
    monkeypatch.setattr(
        "subtitler.ui.main_window.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )
    win = MainWindow(use_fake_asr=True)
    qtbot.addWidget(win)
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"")
    win.open_media(str(media))
    yield win
    win.jobs.cancel()
    win.jobs.wait(5000)


def transcribe(qtbot, win):
    with qtbot.waitSignal(win.jobs.busy_changed, timeout=15000, check_params_cb=lambda b: not b):
        win.transcribe()


def test_full_editing_flow(qtbot, window, tmp_path, monkeypatch):
    win = window
    # Cancel part way through.
    win.transcribe()
    qtbot.waitUntil(lambda: win.progress.value() > jobs.EXTRACTION_SHARE, timeout=5000)
    win.cancel_job()
    qtbot.waitUntil(lambda: not win.jobs.is_busy(), timeout=5000)
    assert win.status_label.text() == "Cancelled"
    assert win.project.cues == []

    # Run to completion.
    transcribe(qtbot, win)
    assert win.project.cues
    assert win.table.cue_model.rowCount() == len(win.project.cues)
    assert win.lint.topLevelItemCount() > 0
    assert win.device_label.text().startswith("Device: fake")

    # Activating a warning seeks and selects its cue.
    group = win.lint.topLevelItem(0)
    violation = group.child(0).data(0, Qt.ItemDataRole.UserRole)
    win.lint.violation_activated.emit(violation)
    assert win.project.cues[win.table.selected_position()].index == violation.cue_index

    # Edit a cue's text through the table model.
    model = win.table.cue_model
    model.setData(model.index(1, 5), "A hand edited | subtitle line")
    qtbot.waitUntil(lambda: win.project.cues[1].locked)
    edited = copy.deepcopy(win.project.cues[1])
    assert edited.lines == ["A hand edited", "subtitle line"]

    # Re-segment keeps the edited cue exactly.
    win.resegment()
    kept = [cue for cue in win.project.cues if cue.locked]
    assert len(kept) == 1
    assert (kept[0].lines, kept[0].start, kept[0].end) == (
        edited.lines,
        edited.start,
        edited.end,
    )

    # Undo the re-segment, then the edit.
    win.undo()
    win.undo()
    assert not any(cue.locked for cue in win.project.cues)
    assert win.project.cues[1].lines != edited.lines

    # Export SRT and read it back.
    out = tmp_path / "out.srt"
    monkeypatch.setattr(
        "subtitler.ui.main_window.QFileDialog.getSaveFileName", lambda *a, **k: (str(out), "")
    )
    win.export_srt()
    exported = read_srt(out)
    assert [cue.lines for cue in exported] == [cue.lines for cue in win.project.cues]


def test_split_and_merge_shortcuts(qtbot, window):
    win = window
    transcribe(qtbot, win)
    count = len(win.project.cues)
    first = win.project.cues[0]
    win.video.position = lambda: (first.start + first.end) / 2
    win.split_at_playhead()
    assert len(win.project.cues) == count + 1
    win.table.select_cue(0)
    win.merge_with_next()
    assert len(win.project.cues) == count
    assert win.project.cues[0].text() == first.text()


def test_detect_shots_updates_project(qtbot, window, monkeypatch):
    win = window
    monkeypatch.setattr(jobs, "detect_shots", lambda path, progress, cancel: [1.0, 5.0])
    with qtbot.waitSignal(win.jobs.busy_changed, timeout=10000, check_params_cb=lambda b: not b):
        win.detect_shots()
    assert win.project.shot_boundaries == [1.0, 5.0]
    assert win.waveform.shots == [1.0, 5.0]
    assert win.status_label.text() == "Found 2 shot changes"
