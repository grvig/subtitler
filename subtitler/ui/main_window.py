"""Main window. Owns the project and the undo stack, and keeps the views in
step with them. All background work goes through JobManager; results come
back as signals and are applied here, on the GUI thread."""

import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from subtitler.core.models import Project
from subtitler.core.rules import RuleEngine
from subtitler.core.rules.profiles import ENGLISH, PROFILES
from subtitler.io import project as project_io
from subtitler.io.srt import read_srt, write_srt
from subtitler.io.vtt import write_vtt
from subtitler.pipeline.asr import FakeASRBackend, FasterWhisperBackend
from subtitler.pipeline.audio import ffmpeg_available
from subtitler.pipeline.jobs import JobManager, ShotDetectionJob, TranscriptionJob
from subtitler.ui.commands import (
    CommandStack,
    EditTextCommand,
    MergeCuesCommand,
    ResegmentCommand,
    SetTimingCommand,
    SplitCueCommand,
    layout_lines,
)
from subtitler.ui.cue_table import CueTable
from subtitler.ui.lint_panel import LintPanel
from subtitler.ui.video_pane import VideoPane
from subtitler.ui.waveform import WaveformView

MODEL_SIZES = ["small", "medium"]
PROJECT_SUFFIX = ".subproj"
MEDIA_FILTER = "Media (*.mp4 *.mkv *.mov *.avi *.webm *.mp3 *.wav *.m4a *.flac);;All files (*)"


class MainWindow(QMainWindow):
    def __init__(self, use_fake_asr: bool = False):
        super().__init__()
        self.setWindowTitle("Subtitler")
        self.resize(1400, 900)
        self.project = Project(profile=ENGLISH.name)
        self.project_path = None
        self.profile = ENGLISH
        self.use_fake_asr = use_fake_asr
        self.backends = {}
        self.violations = []
        self.stack = CommandStack()
        self.stack.listeners.append(self.refresh)
        self.jobs = JobManager(self)
        self.jobs.busy_changed.connect(self._on_busy)

        self.video = VideoPane()
        self.waveform = WaveformView()
        self.table = CueTable()
        self.lint = LintPanel()
        self._build_layout()
        self._build_actions()
        self._build_status_bar()
        self._connect_views()

        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(project_io.AUTOSAVE_INTERVAL_SECONDS * 1000)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start()
        self.refresh()

    # ----- construction -------------------------------------------------

    def _build_layout(self) -> None:
        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self.video)
        top.addWidget(self.lint)
        top.setStretchFactor(0, 3)
        top.setStretchFactor(1, 2)
        main = QSplitter(Qt.Orientation.Vertical)
        main.addWidget(top)
        main.addWidget(self.waveform)
        main.addWidget(self.table)
        main.setStretchFactor(0, 5)
        main.setStretchFactor(1, 0)
        main.setStretchFactor(2, 4)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(main)
        self.setCentralWidget(container)

    def _action(self, text, slot, shortcut=None) -> QAction:
        action = QAction(text, self)
        action.triggered.connect(slot)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        # Window-wide, so the shortcuts work whichever pane has focus.
        action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.addAction(action)
        return action

    def _build_actions(self) -> None:
        self.open_action = self._action("Open Media...", self.open_media, "Ctrl+O")
        self.import_action = self._action("Import Subtitles...", self.import_subtitles)
        self.open_project_action = self._action("Open Project...", self.open_project)
        self.save_action = self._action("Save Project", self.save_project, "Ctrl+S")
        self.export_srt_action = self._action("Export SRT...", self.export_srt, "Ctrl+E")
        self.export_vtt_action = self._action("Export VTT...", self.export_vtt)
        self.transcribe_action = self._action("Transcribe", self.transcribe, "Ctrl+T")
        self.resegment_action = self._action("Re-segment", self.resegment, "Ctrl+R")
        self.shots_action = self._action("Detect Shot Changes", self.detect_shots, "Ctrl+D")
        self.undo_action = self._action("Undo", self.undo, "Ctrl+Z")
        self.redo_action = self._action("Redo", self.redo, "Ctrl+Y")
        self.split_action = self._action("Split at Playhead", self.split_at_playhead, "Ctrl+Return")
        self.merge_action = self._action("Merge with Next", self.merge_with_next, "Ctrl+M")
        self.play_action = self._action("Play / Pause", self.video.toggle_play, "Space")
        self.in_action = self._action("Set In-Point", self.set_in_point, "I")
        self.out_action = self._action("Set Out-Point", self.set_out_point, "O")

        menus = self.menuBar()
        file_menu = menus.addMenu("&File")
        for action in [
            self.open_action,
            self.import_action,
            self.open_project_action,
            self.save_action,
            None,
            self.export_srt_action,
            self.export_vtt_action,
        ]:
            if action is None:
                file_menu.addSeparator()
            else:
                file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)
        edit_menu = menus.addMenu("&Edit")
        for action in [self.undo_action, self.redo_action, self.split_action, self.merge_action]:
            edit_menu.addAction(action)
        edit_menu.addAction(self.in_action)
        edit_menu.addAction(self.out_action)
        tools_menu = menus.addMenu("&Subtitles")
        tools_menu.addAction(self.transcribe_action)
        tools_menu.addAction(self.shots_action)
        tools_menu.addAction(self.resegment_action)
        playback_menu = menus.addMenu("&Playback")
        playback_menu.addAction(self.play_action)

        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        for action in [
            self.open_action,
            self.import_action,
            self.transcribe_action,
            self.shots_action,
            self.resegment_action,
            self.export_srt_action,
            self.export_vtt_action,
            self.undo_action,
            self.redo_action,
        ]:
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" Profile: "))
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(list(PROFILES))
        self.profile_combo.currentTextChanged.connect(self.change_profile)
        toolbar.addWidget(self.profile_combo)
        toolbar.addWidget(QLabel("  Model: "))
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_SIZES)
        self.model_combo.setToolTip("small fits in 4 GB of VRAM; medium is slower and may not")
        toolbar.addWidget(self.model_combo)

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        self.device_label = QLabel("Device: not loaded")
        self.status_label = QLabel("Ready")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setVisible(False)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self.cancel_job)
        bar.addWidget(self.status_label, 1)
        bar.addPermanentWidget(self.progress)
        bar.addPermanentWidget(self.cancel_button)
        bar.addPermanentWidget(self.device_label)

    def _connect_views(self) -> None:
        self.video.position_changed.connect(self.waveform.set_playhead)
        self.waveform.seek_requested.connect(self.video.seek)
        self.waveform.timing_changed.connect(self._on_timing_dragged)
        self.table.cue_selected.connect(self._on_cue_selected)
        self.table.cue_model.text_edited.connect(self._on_text_edited)
        self.lint.violation_activated.connect(self._on_violation)

    # ----- keeping views in step ----------------------------------------

    def refresh(self) -> None:
        cues = self.project.cues
        self.violations = RuleEngine().validate(cues, self.profile, self.project.shot_boundaries)
        selected = self.table.selected_position()
        self.table.cue_model.set_cues(cues, self.profile)
        selected = min(selected, len(cues) - 1)
        self.table.select_cue(selected)
        self.waveform.set_selected(selected)
        self.waveform.set_cues(cues)
        self.waveform.set_shots(self.project.shot_boundaries)
        self.video.set_cues(cues)
        self.lint.set_violations(self.violations)
        busy = self.jobs.is_busy()
        has_media = self.project.media_path is not None
        self.undo_action.setEnabled(self.stack.can_undo() and not busy)
        self.redo_action.setEnabled(self.stack.can_redo() and not busy)
        self.transcribe_action.setEnabled(has_media and not busy and ffmpeg_available())
        self.shots_action.setEnabled(has_media and not busy)
        self.resegment_action.setEnabled(bool(cues) and not busy)
        for action in [self.export_srt_action, self.export_vtt_action, self.save_action]:
            action.setEnabled(bool(cues))
        title = "Subtitler"
        if has_media:
            title = f"{Path(self.project.media_path).name} - Subtitler"
        if self.project.dirty:
            title = f"*{title}"
        self.setWindowTitle(title)

    def _on_cue_selected(self, position: int) -> None:
        self.waveform.set_selected(position)
        self.video.seek(self.project.cues[position].start)

    def _on_violation(self, violation) -> None:
        self.video.seek(violation.timecode)
        for position, cue in enumerate(self.project.cues):
            if cue.index == violation.cue_index:
                self.table.select_cue(position)
                self.waveform.set_selected(position)
                break

    def _on_busy(self, busy: bool) -> None:
        self.progress.setVisible(busy)
        self.cancel_button.setVisible(busy)
        self.cancel_button.setEnabled(True)
        self.refresh()

    # ----- files ---------------------------------------------------------

    def open_media(self, path: str | None = None) -> None:
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open media", "", MEDIA_FILTER)
        if not path:
            return
        if not self._confirm_discard():
            return
        self.project = Project(media_path=path, profile=self.profile.name)
        self.project_path = path + PROJECT_SUFFIX
        self.stack.clear()
        self.video.load(path)
        self.waveform.set_peaks(np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), 100)
        self.status_label.setText(f"Opened {Path(path).name}")
        if os.path.exists(self.project_path) or project_io.find_recovery(self.project_path):
            self._load_project_file(self.project_path)
        self.refresh()

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open project", "", f"Subtitler project (*{PROJECT_SUFFIX})"
        )
        if path and self._confirm_discard():
            self._load_project_file(path)
            if self.project.media_path:
                self.video.load(self.project.media_path)
            self.refresh()

    def _load_project_file(self, path: str) -> None:
        source = path
        recovery = project_io.find_recovery(path)
        if recovery is not None:
            answer = QMessageBox.question(
                self,
                "Recover unsaved work",
                "An autosaved copy newer than the saved project was found.\n"
                "The last session probably ended without saving.\n\n"
                "Open the autosaved copy?",
            )
            if answer == QMessageBox.StandardButton.Yes:
                source = recovery
        if not os.path.exists(source):
            return
        self.project = project_io.load_project(source)
        self.project.dirty = source != path
        self.project_path = path
        peaks = project_io.load_peaks(path)
        if peaks is not None:
            self.waveform.set_peaks(*peaks)
        if self.project.profile in PROFILES:
            self.profile_combo.setCurrentText(self.project.profile)
        self.stack.clear()

    def save_project(self) -> bool:
        if self.project_path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save project", "", f"Subtitler project (*{PROJECT_SUFFIX})"
            )
            if not path:
                return False
            self.project_path = path
        project_io.save_project(self.project, self.project_path)
        project_io.discard_autosave(self.project_path)
        self.project.dirty = False
        self.status_label.setText(f"Saved {Path(self.project_path).name}")
        self.refresh()
        return True

    def autosave(self) -> None:
        if self.project_path and self.project.dirty and self.project.cues:
            project_io.write_autosave(self.project, self.project_path)

    def import_subtitles(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import subtitles", "", "SubRip (*.srt)")
        if not path:
            return
        cues = read_srt(path)
        if not cues:
            QMessageBox.warning(self, "Import subtitles", "No cues could be read from that file.")
            return
        self.project.cues = cues
        self.project.dirty = True
        self.stack.clear()
        self.status_label.setText(f"Imported {len(cues)} cues")

    def _export(self, suffix: str, writer) -> None:
        default = ""
        if self.project.media_path:
            default = str(Path(self.project.media_path).with_suffix(suffix))
        path, _ = QFileDialog.getSaveFileName(self, f"Export {suffix}", default, f"*{suffix}")
        if path:
            writer(path, self.project.cues)
            self.status_label.setText(f"Exported {len(self.project.cues)} cues to {path}")

    def export_srt(self) -> None:
        self._export(".srt", write_srt)

    def export_vtt(self) -> None:
        self._export(".vtt", write_vtt)

    def _confirm_discard(self) -> bool:
        if not self.project.dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "The current subtitles have unsaved changes. Discard them?",
        )
        return answer == QMessageBox.StandardButton.Yes

    # ----- transcription -------------------------------------------------

    def _backend(self):
        if self.use_fake_asr:
            return self.backends.setdefault("fake", FakeASRBackend(delay=0.3))
        size = self.model_combo.currentText()
        # One backend per model size, so a loaded model is reused between runs.
        return self.backends.setdefault(size, FasterWhisperBackend(model_size=size))

    def transcribe(self) -> None:
        if self.project.media_path is None or self.jobs.is_busy():
            return
        if self.project.cues and not self._confirm_replace():
            return
        job = TranscriptionJob(
            self.project.media_path, self._backend(), self.profile, self.project.shot_boundaries
        )
        self._start_job(job, self._on_transcribed)

    def _start_job(self, job, on_finished) -> None:
        job.signals.progress.connect(self.progress.setValue)
        job.signals.status.connect(self.status_label.setText)
        job.signals.finished.connect(on_finished)
        job.signals.failed.connect(self._on_job_failed)
        job.signals.cancelled.connect(self._on_job_cancelled)
        self.progress.setValue(0)
        self.jobs.start(job)

    def detect_shots(self) -> None:
        if self.project.media_path is None or self.jobs.is_busy():
            return
        self._start_job(ShotDetectionJob(self.project.media_path), self._on_shots)

    def _on_shots(self, cuts: list[float]) -> None:
        self.project.shot_boundaries = cuts
        self.project.dirty = True
        message = f"Found {len(cuts)} shot changes"
        if self.project.cues:
            message = f"{message}. Re-segment to fit the cues to them."
        self.status_label.setText(message)
        self.refresh()

    def _confirm_replace(self) -> bool:
        answer = QMessageBox.question(
            self, "Transcribe", "Replace the current cues with a new transcription?"
        )
        return answer == QMessageBox.StandardButton.Yes

    def cancel_job(self) -> None:
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Cancelling...")
        self.jobs.cancel()

    def _on_transcribed(self, result) -> None:
        self.project.cues = result.cues
        self.project.dirty = True
        self.stack.clear()
        self.waveform.set_peaks(result.peaks_min, result.peaks_max, result.buckets_per_second)
        if self.project_path:
            project_io.save_peaks(
                self.project_path, result.peaks_min, result.peaks_max, result.buckets_per_second
            )
        device = f"Device: {result.device}"
        language = getattr(
            self.backends.get(self.model_combo.currentText()), "detected_language", None
        )
        if language and not self.use_fake_asr:
            device = f"{device}  |  language {language}"
        self.device_label.setText(device)
        if result.fallback_reason:
            self.device_label.setToolTip(f"GPU not used: {result.fallback_reason}")
        self.status_label.setText(
            f"Transcribed: {len(result.cues)} cues, {len(result.violations)} warnings"
        )
        self.refresh()

    def _on_job_failed(self, message: str) -> None:
        self.status_label.setText("Failed")
        QMessageBox.critical(self, "Job failed", message)

    def _on_job_cancelled(self) -> None:
        self.status_label.setText("Cancelled")

    # ----- editing -------------------------------------------------------

    def _run(self, command_factory) -> None:
        try:
            command = command_factory()
        except ValueError as error:
            self.status_label.setText(str(error))
            return
        self.stack.execute(command)

    def _on_text_edited(self, position: int, text: str) -> None:
        lines = layout_lines(text, self.profile)
        # Deferred so the table's editor has closed before the model is reset.
        QTimer.singleShot(
            0, lambda: self._run(lambda: EditTextCommand(self.project, position, lines))
        )

    def _on_timing_dragged(self, position: int, start: float, end: float) -> None:
        self._run(lambda: SetTimingCommand(self.project, position, start, end))

    def _position_for_edit(self) -> int:
        """The cue under the playhead, or else the selected cue."""
        now = self.video.position()
        for position, cue in enumerate(self.project.cues):
            if cue.start <= now < cue.end:
                return position
        return self.table.selected_position()

    def split_at_playhead(self) -> None:
        position = self._position_for_edit()
        if position < 0:
            return
        now = self.video.position()
        self._run(lambda: SplitCueCommand(self.project, position, now, self.profile))

    def merge_with_next(self) -> None:
        position = self.table.selected_position()
        if position < 0:
            return
        self._run(lambda: MergeCuesCommand(self.project, position, self.profile))

    def set_in_point(self) -> None:
        position = self.table.selected_position()
        if position < 0:
            return
        cue = self.project.cues[position]
        now = self.video.position()
        if now >= cue.end:
            self.status_label.setText("In-point must be before the cue's end")
            return
        self._run(lambda: SetTimingCommand(self.project, position, now, cue.end))

    def set_out_point(self) -> None:
        position = self.table.selected_position()
        if position < 0:
            return
        cue = self.project.cues[position]
        now = self.video.position()
        if now <= cue.start:
            self.status_label.setText("Out-point must be after the cue's start")
            return
        self._run(lambda: SetTimingCommand(self.project, position, cue.start, now))

    def resegment(self) -> None:
        if not any(cue.words for cue in self.project.cues):
            QMessageBox.information(
                self,
                "Re-segment",
                "These cues have no word timings (they were imported), so they cannot be "
                "re-segmented. Transcribe the media to get word timings.",
            )
            return
        self._run(lambda: ResegmentCommand(self.project, self.profile))
        self.status_label.setText(f"Re-segmented into {len(self.project.cues)} cues")

    def undo(self) -> None:
        self.stack.undo()

    def redo(self) -> None:
        self.stack.redo()

    def change_profile(self, name: str) -> None:
        self.profile = PROFILES[name]
        self.project.profile = name
        self.refresh()

    # ----- shutdown ------------------------------------------------------

    def closeEvent(self, event) -> None:
        if not self._confirm_discard():
            event.ignore()
            return
        self.jobs.cancel()
        self.jobs.wait(10000)
        event.accept()
