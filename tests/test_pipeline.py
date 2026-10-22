import threading
import wave

import numpy as np
import pytest

from subtitler.core.models import Word
from subtitler.core.rules.profiles import ENGLISH
from subtitler.pipeline import audio, jobs
from subtitler.pipeline.asr import FakeASRBackend, FasterWhisperBackend, join_split_words


def write_tone(path, seconds=1.0, rate=16000):
    t = np.arange(int(seconds * rate)) / rate
    samples = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


def test_compute_peaks(tmp_path):
    path = tmp_path / "tone.wav"
    write_tone(path, seconds=1.5)
    mins, maxs = audio.compute_peaks(path, buckets_per_second=100)
    assert mins.shape == maxs.shape == (150,)
    assert maxs.max() == pytest.approx(0.5, abs=0.01)
    assert mins.min() == pytest.approx(-0.5, abs=0.01)


def test_missing_tool_raises_ffmpeg_error():
    with pytest.raises(audio.FFmpegError):
        audio._run(["definitely-not-a-real-program-xyz"])


def test_fake_backend_reports_progress_and_cancels():
    backend = FakeASRBackend()
    seen = []
    words = backend.transcribe("unused.wav", seen.append, threading.Event())
    assert words == backend.words
    assert seen[-1] == 100.0
    cancel = threading.Event()
    cancel.set()
    assert backend.transcribe("unused.wav", None, cancel) == []


def test_split_hyphenated_words_are_joined():
    words = [
        Word("a", 0.0, 0.2),
        Word("Co", 0.3, 0.5),
        Word("-Pilot", 0.5, 0.9, 0.7),
        Word("long", 1.0, 1.2),
        Word("-term", 1.2, 1.5),
        Word("-", 1.6, 1.7),
        Word("done.", 1.8, 2.0),
    ]
    joined = join_split_words(words)
    assert [w.text for w in joined] == ["a", "Co-Pilot", "long-term", "-", "done."]
    assert (joined[1].start, joined[1].end, joined[1].confidence) == (0.3, 0.9, 0.7)


def test_whisper_backend_does_not_load_on_construction():
    backend = FasterWhisperBackend()
    assert backend._model is None
    assert backend.model_size == "small"
    assert backend.compute_type == "int8_float16"


@pytest.fixture
def fake_extraction(monkeypatch):
    def fake_extract(media_path, out_path):
        write_tone(out_path, seconds=20.0)

    monkeypatch.setattr(jobs, "extract_audio", fake_extract)


def test_transcription_job_delivers_result(qtbot, fake_extraction):
    job = jobs.TranscriptionJob("clip.mp4", FakeASRBackend(), ENGLISH)
    progress = []
    job.signals.progress.connect(progress.append)
    manager = jobs.JobManager()
    with qtbot.waitSignal(job.signals.finished, timeout=10000) as blocker:
        assert manager.start(job)
    result = blocker.args[0]
    assert result.cues
    assert result.device == "fake"
    assert result.peaks_min.size == 2000
    assert progress == sorted(progress)
    assert progress[-1] == 100
    qtbot.waitUntil(lambda: not manager.is_busy())


def test_transcription_job_can_be_cancelled(qtbot, fake_extraction):
    job = jobs.TranscriptionJob("clip.mp4", FakeASRBackend(delay=0.2), ENGLISH)
    progress = []
    job.signals.progress.connect(progress.append)
    finished = []
    job.signals.finished.connect(finished.append)
    manager = jobs.JobManager()
    manager.start(job)
    # Cancel once transcription is under way, not before it starts.
    qtbot.waitUntil(lambda: bool(progress) and progress[-1] > jobs.EXTRACTION_SHARE, timeout=5000)
    with qtbot.waitSignal(job.signals.cancelled, timeout=10000):
        manager.cancel()
    qtbot.waitUntil(lambda: not manager.is_busy())
    assert finished == []
    assert progress[-1] < jobs.EXTRACTION_SHARE + jobs.ASR_SHARE


def test_manager_runs_one_job_at_a_time(qtbot, fake_extraction):
    manager = jobs.JobManager()
    first = jobs.TranscriptionJob("a.mp4", FakeASRBackend(delay=0.05), ENGLISH)
    second = jobs.TranscriptionJob("b.mp4", FakeASRBackend(), ENGLISH)
    with qtbot.waitSignal(first.signals.finished, timeout=10000):
        assert manager.start(first)
        assert not manager.start(second)
