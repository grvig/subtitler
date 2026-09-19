"""Audio extraction and waveform peaks, using the ffmpeg command line tools."""

import shutil
import subprocess
import wave

import numpy as np

TARGET_SAMPLE_RATE = 16000

# Keeps a console window from flashing up on Windows when run from the GUI.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class FFmpegError(Exception):
    """ffmpeg or ffprobe failed. The tool's stderr output is kept in .stderr."""

    def __init__(self, message: str, stderr: str = ""):
        super().__init__(message)
        self.stderr = stderr


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _run(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=_NO_WINDOW,
        )
    except FileNotFoundError as error:
        raise FFmpegError(f"{command[0]} was not found on PATH") from error
    if result.returncode != 0:
        detail = result.stderr.strip()
        raise FFmpegError(f"{command[0]} exited with code {result.returncode}: {detail}", detail)
    return result.stdout


def extract_audio(media_path, out_path) -> None:
    """Decode the media file's audio to a 16 kHz mono 16-bit WAV file."""
    _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(media_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(out_path),
        ]
    )


def probe_duration(media_path) -> float:
    output = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ]
    )
    try:
        return float(output.strip())
    except ValueError as error:
        raise FFmpegError(f"ffprobe gave no duration for {media_path}", output) from error


def compute_peaks(wav_path, buckets_per_second: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Min and max sample value per bucket, scaled to -1..1, for drawing a
    waveform. Expects the 16-bit mono WAV written by extract_audio."""
    with wave.open(str(wav_path), "rb") as handle:
        rate = handle.getframerate()
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    bucket = max(1, rate // buckets_per_second)
    count = -(-samples.size // bucket)
    padded = np.zeros(count * bucket, dtype=np.float32)
    padded[: samples.size] = samples / 32768.0
    blocks = padded.reshape(count, bucket)
    return blocks.min(axis=1), blocks.max(axis=1)
