"""Background jobs.

A job runs on a QThreadPool worker thread and must never touch a widget.
Everything it produces goes back to the GUI thread through JobSignals.
The signals object is created on the GUI thread, so Qt queues each emit
into the GUI thread's event loop. The work itself is done by the plain
functions in pipeline/run.py.
"""

import threading

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from subtitler.core.models import Cue
from subtitler.core.rules.profiles import StyleProfile
from subtitler.pipeline import run
from subtitler.pipeline.asr import ASRBackend
from subtitler.pipeline.shots import detect_shots


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

    def execute(self):
        raise NotImplementedError

    def run(self) -> None:
        try:
            result = self.execute()
            if self.cancel_event.is_set():
                raise run.Cancelled()
        except run.Cancelled:
            self.signals.cancelled.emit()
        except Exception as error:
            self.signals.failed.emit(f"{type(error).__name__}: {error}")
        else:
            self.signals.finished.emit(result)

    def _hooks(self) -> dict:
        return {
            "progress": self.signals.progress.emit,
            "status": self.signals.status.emit,
            "cancel_event": self.cancel_event,
        }


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

    def execute(self) -> run.TranscriptionResult:
        return run.transcribe_media(
            self.media_path, self.backend, self.profile, self.shot_boundaries, **self._hooks()
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
        self.signals.progress.emit(100)
        return cuts


class RealignJob(Job):
    """Gives each cue's text new word timings; see run.realign_cues."""

    def __init__(self, media_path, backend: ASRBackend, cues: list[tuple[int, Cue]]):
        super().__init__()
        self.media_path = media_path
        self.backend = backend
        self.cues = cues

    def execute(self):
        return run.realign_cues(self.media_path, self.backend, self.cues, **self._hooks())


class BatchJob(Job):
    """Transcribes every file in a list and writes subtitles beside each one."""

    def __init__(self, paths, backend: ASRBackend, profile: StyleProfile, find_shots: bool):
        super().__init__()
        self.paths = paths
        self.backend = backend
        self.profile = profile
        self.find_shots = find_shots

    def execute(self) -> list[run.BatchItem]:
        return run.batch_transcribe(
            self.paths, self.backend, self.profile, find_shots=self.find_shots, **self._hooks()
        )


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
