"""Background jobs.

A job runs on a QThreadPool worker thread and must never touch a widget.
Everything it produces goes back to the GUI thread through JobSignals.
The signals object is created on the GUI thread, so Qt queues each emit
into the GUI thread's event loop.
"""

import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from subtitler.core.models import Cue, Violation, Word
from subtitler.core.rules import RuleEngine
from subtitler.core.rules.profiles import StyleProfile
from subtitler.core.segmentation import segment
from subtitler.pipeline.asr import ASRBackend
from subtitler.pipeline.audio import compute_peaks, extract_audio
from subtitler.pipeline.shots import detect_shots

PEAK_BUCKETS_PER_SECOND = 100

# How the progress bar is shared between the stages, in percent.
EXTRACTION_SHARE = 15
ASR_SHARE = 70


class JobCancelled(Exception):
    pass


class JobSignals(QObject):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()


class Job(QRunnable):
    def __init__(self):
        super().__init__()
        # The manager keeps the reference, so Qt must not delete the runnable.
        self.setAutoDelete(False)
        self.signals = JobSignals()
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise JobCancelled()

    def execute(self):
        raise NotImplementedError

    def run(self) -> None:
        try:
            result = self.execute()
            self.check_cancelled()
        except JobCancelled:
            self.signals.cancelled.emit()
        except Exception as error:
            self.signals.failed.emit(f"{type(error).__name__}: {error}")
        else:
            self.signals.finished.emit(result)


@dataclass
class TranscriptionResult:
    words: list[Word]
    cues: list[Cue]
    violations: list[Violation]
    peaks_min: np.ndarray
    peaks_max: np.ndarray
    buckets_per_second: int
    device: str
    fallback_reason: str | None = None
    shot_boundaries: list[float] = field(default_factory=list)


class TranscriptionJob(Job):
    def __init__(
        self,
        media_path,
        backend: ASRBackend,
        profile: StyleProfile,
        shot_boundaries: list[float] | None = None,
    ):
        super().__init__()
        self.media_path = media_path
        self.backend = backend
        self.profile = profile
        self.shot_boundaries = list(shot_boundaries or [])

    def _asr_progress(self, percent: float) -> None:
        self.signals.progress.emit(EXTRACTION_SHARE + int(percent * ASR_SHARE / 100))

    def execute(self) -> TranscriptionResult:
        with tempfile.TemporaryDirectory(prefix="subtitler-") as work_dir:
            wav_path = Path(work_dir) / "audio.wav"
            self.signals.status.emit("Extracting audio")
            self.signals.progress.emit(0)
            extract_audio(self.media_path, wav_path)
            self.check_cancelled()
            peaks_min, peaks_max = compute_peaks(wav_path, PEAK_BUCKETS_PER_SECOND)
            self.signals.progress.emit(EXTRACTION_SHARE)
            self.signals.status.emit("Loading model and transcribing")
            words = self.backend.transcribe(wav_path, self._asr_progress, self.cancel_event)
            self.check_cancelled()
        self.signals.status.emit("Segmenting")
        self.signals.progress.emit(EXTRACTION_SHARE + ASR_SHARE)
        cues = segment(words, self.shot_boundaries, self.profile)
        violations = RuleEngine().validate(cues, self.profile, self.shot_boundaries)
        self.signals.progress.emit(100)
        self.signals.status.emit("Done")
        return TranscriptionResult(
            words=words,
            cues=cues,
            violations=violations,
            peaks_min=peaks_min,
            peaks_max=peaks_max,
            buckets_per_second=PEAK_BUCKETS_PER_SECOND,
            device=self.backend.device_used,
            fallback_reason=getattr(self.backend, "fallback_reason", None),
            shot_boundaries=self.shot_boundaries,
        )


class ShotDetectionJob(Job):
    """Finds shot changes in the video. The result is a list of times."""

    def __init__(self, media_path):
        super().__init__()
        self.media_path = media_path

    def execute(self) -> list[float]:
        self.signals.status.emit("Detecting shot changes")
        self.signals.progress.emit(0)
        cuts = detect_shots(
            self.media_path, lambda p: self.signals.progress.emit(int(p)), self.cancel_event
        )
        self.check_cancelled()
        self.signals.progress.emit(100)
        return cuts


class JobManager(QObject):
    """Runs at most one heavy job at a time."""

    busy_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.current = None

    def is_busy(self) -> bool:
        return self.current is not None

    def start(self, job: Job) -> bool:
        if self.current is not None:
            return False
        self.current = job
        job.signals.finished.connect(self._clear)
        job.signals.failed.connect(self._clear)
        job.signals.cancelled.connect(self._clear)
        self.busy_changed.emit(True)
        self.pool.start(job)
        return True

    def cancel(self) -> None:
        if self.current is not None:
            self.current.cancel()

    def wait(self, milliseconds: int = -1) -> bool:
        return self.pool.waitForDone(milliseconds)

    def _clear(self, *args) -> None:
        self.current = None
        self.busy_changed.emit(False)
