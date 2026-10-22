"""The processing pipeline as plain functions, with no Qt.

The GUI runs these through the jobs in pipeline/jobs.py; the batch command
line tool calls them directly. Progress is reported as an integer percentage
through an optional callback, status messages through another, and a
threading.Event can be set to stop early, which raises Cancelled.
"""

import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from subtitler.core.alignment import align_words
from subtitler.core.models import Cue, Violation, Word
from subtitler.core.rules import RuleEngine
from subtitler.core.rules.profiles import StyleProfile
from subtitler.core.segmentation import segment
from subtitler.io.srt import write_srt
from subtitler.io.vtt import write_vtt
from subtitler.pipeline.asr import ASRBackend
from subtitler.pipeline.audio import compute_peaks, extract_audio
from subtitler.pipeline.shots import detect_shots

PEAK_BUCKETS_PER_SECOND = 100
# Audio taken on each side of a cue when it is transcribed again on its own.
ALIGN_PADDING = 0.25

# How the progress bar is shared between the stages, in percent. When shot
# changes are detected as well, their share comes out of the ASR share.
EXTRACTION_SHARE = 15
ASR_SHARE = 70
SHOT_SHARE = 15

MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mp3", ".wav", ".m4a", ".flac", ".ogg",
}  # fmt: skip
WRITERS = {"srt": write_srt, "vtt": write_vtt}

Progress = Callable[[int], None] | None
Status = Callable[[str], None] | None


class Cancelled(Exception):
    pass


def _report(callback, value) -> None:
    if callback is not None:
        callback(value)


def _check(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled()


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


def transcribe_media(
    media_path,
    backend: ASRBackend,
    profile: StyleProfile,
    shot_boundaries: list[float] | None = None,
    find_shots: bool = False,
    progress: Progress = None,
    status: Status = None,
    cancel_event: threading.Event | None = None,
) -> TranscriptionResult:
    """Extract the audio, optionally find shot changes, recognise speech,
    segment it into cues and check them."""
    shots = list(shot_boundaries or [])
    shot_share = 0
    if find_shots:
        shot_share = SHOT_SHARE
    asr_share = ASR_SHARE - shot_share
    with tempfile.TemporaryDirectory(prefix="subtitler-") as work_dir:
        wav_path = Path(work_dir) / "audio.wav"
        _report(status, "Extracting audio")
        _report(progress, 0)
        extract_audio(media_path, wav_path)
        _check(cancel_event)
        peaks_min, peaks_max = compute_peaks(wav_path, PEAK_BUCKETS_PER_SECOND)
        _report(progress, EXTRACTION_SHARE)
        if find_shots:
            _report(status, "Detecting shot changes")
            shots = detect_shots(
                media_path,
                lambda p: _report(progress, EXTRACTION_SHARE + int(p * shot_share / 100)),
                cancel_event,
            )
            _check(cancel_event)
        asr_start = EXTRACTION_SHARE + shot_share
        _report(status, "Loading model and transcribing")
        words = backend.transcribe(
            wav_path,
            lambda p: _report(progress, asr_start + int(p * asr_share / 100)),
            cancel_event,
        )
        _check(cancel_event)
    _report(status, "Segmenting")
    _report(progress, asr_start + asr_share)
    cues = segment(words, shots, profile)
    violations = RuleEngine().validate(cues, profile, shots)
    _report(progress, 100)
    _report(status, "Done")
    return TranscriptionResult(
        words=words,
        cues=cues,
        violations=violations,
        peaks_min=peaks_min,
        peaks_max=peaks_max,
        buckets_per_second=PEAK_BUCKETS_PER_SECOND,
        device=backend.device_used,
        fallback_reason=getattr(backend, "fallback_reason", None),
        shot_boundaries=shots,
    )


def realign_cues(
    media_path,
    backend: ASRBackend,
    cues: list[tuple[int, Cue]],
    progress: Progress = None,
    status: Status = None,
    cancel_event: threading.Event | None = None,
) -> list[tuple[int, float, list[Word]]]:
    """Transcribe the audio under each cue again, with the cue's text as a
    hint, and work out word timings for that text. Returns (position,
    original start, words) for each cue."""
    results = []
    with tempfile.TemporaryDirectory(prefix="subtitler-") as work_dir:
        wav_path = Path(work_dir) / "cue.wav"
        for number, (position, cue) in enumerate(cues):
            _check(cancel_event)
            _report(status, f"Realigning cue {cue.index}")
            offset = max(0.0, cue.start - ALIGN_PADDING)
            extract_audio(media_path, wav_path, offset, cue.end + ALIGN_PADDING - offset)
            heard = backend.transcribe(wav_path, None, cancel_event, cue.text())
            # Back onto the media's clock, and inside the cue.
            inside = []
            for word in heard:
                start = min(max(word.start + offset, cue.start), cue.end)
                end = min(max(word.end + offset, start), cue.end)
                inside.append(Word(word.text, start, end, word.confidence))
            words = align_words(cue.text(), inside, cue.start, cue.end)
            results.append((position, cue.start, words))
            _report(progress, int(100 * (number + 1) / len(cues)))
    _report(status, "Done")
    return results


def find_media(folder, recursive: bool = False) -> list[Path]:
    pattern = "*"
    if recursive:
        pattern = "**/*"
    found = [p for p in Path(folder).glob(pattern) if p.suffix.lower() in MEDIA_EXTENSIONS]
    return sorted(p for p in found if p.is_file())


@dataclass
class BatchItem:
    media_path: Path
    outputs: list[Path]
    cue_count: int = 0
    violation_count: int = 0
    skipped: bool = False
    error: str | None = None


def output_path(media_path: Path, out_dir, extension: str) -> Path:
    folder = media_path.parent
    if out_dir is not None:
        folder = Path(out_dir)
    return folder / f"{media_path.stem}.{extension}"


def batch_transcribe(
    paths: list[Path],
    backend: ASRBackend,
    profile: StyleProfile,
    out_dir=None,
    formats: tuple[str, ...] = ("srt",),
    find_shots: bool = False,
    overwrite: bool = False,
    progress: Progress = None,
    status: Status = None,
    cancel_event: threading.Event | None = None,
) -> list[BatchItem]:
    """Transcribe each file and write its subtitles. A file that fails is
    recorded and the batch carries on with the next one."""
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    items = []
    for number, media_path in enumerate(paths):
        _check(cancel_event)
        targets = [output_path(media_path, out_dir, extension) for extension in formats]
        if not overwrite and all(target.exists() for target in targets):
            items.append(BatchItem(media_path, targets, skipped=True))
            continue
        _report(status, f"[{number + 1}/{len(paths)}] {media_path.name}")

        def file_progress(percent, number=number):
            _report(progress, int(100 * (number + percent / 100) / len(paths)))

        try:
            result = transcribe_media(
                media_path,
                backend,
                profile,
                find_shots=find_shots,
                progress=file_progress,
                cancel_event=cancel_event,
            )
        except Cancelled:
            raise
        except Exception as error:
            items.append(BatchItem(media_path, [], error=f"{type(error).__name__}: {error}"))
            continue
        for target, extension in zip(targets, formats, strict=True):
            WRITERS[extension](target, result.cues)
        items.append(BatchItem(media_path, targets, len(result.cues), len(result.violations)))
    _report(progress, 100)
    return items
