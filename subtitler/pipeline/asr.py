"""Speech recognition backends.

The rest of the application only sees ASRBackend, so the recogniser can be
swapped, and tests can use FakeASRBackend without downloading a model.
"""

import importlib.util
import os
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from subtitler.core.models import Word

ProgressCallback = Callable[[float], None]

DEFAULT_MODEL_SIZE = "small"
GPU_COMPUTE_TYPE = "int8_float16"
CPU_COMPUTE_TYPE = "int8"
CPU_THREADS = 8
HYPHENS = "-‐‑"


def _register_cuda_libraries() -> None:
    """Make the CUDA DLLs from pip's nvidia-cublas-cu12 and nvidia-cudnn-cu12
    wheels loadable. They sit in site-packages/nvidia/*/bin, which Windows
    does not search on its own. Does nothing if the wheels are not installed."""
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not hasattr(os, "add_dll_directory"):
        return
    for root in spec.submodule_search_locations:
        for bin_dir in Path(root).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


class ASRBackend(ABC):
    # Human readable description of where the model runs, e.g. "cuda (int8_float16)".
    device_used = "not loaded"

    @abstractmethod
    def transcribe(
        self,
        audio_path,
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[Word]:
        """Return timed words. Progress is reported as a percentage from 0 to
        100. If cancel_event is set, stop early and return the words so far."""

    @abstractmethod
    def is_available(self) -> bool:
        """True if the backend can run on this machine."""


class FasterWhisperBackend(ASRBackend):
    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: str = "auto",
        compute_type: str = GPU_COMPUTE_TYPE,
        language: str | None = None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.fallback_reason = None
        self.detected_language = None
        self._model = None

    def is_available(self) -> bool:
        return importlib.util.find_spec("faster_whisper") is not None

    def _load_cpu(self):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model_size, device="cpu", compute_type=CPU_COMPUTE_TYPE, cpu_threads=CPU_THREADS
        )
        self.device_used = f"cpu ({CPU_COMPUTE_TYPE})"
        return self._model

    def _load(self):
        if self._model is not None:
            return self._model
        # Imported here so that importing this module never loads the library.
        _register_cuda_libraries()
        import ctranslate2
        from faster_whisper import WhisperModel

        if self.device == "cpu":
            return self._load_cpu()
        try:
            if ctranslate2.get_cuda_device_count() == 0:
                raise RuntimeError("no CUDA device found")
            self._model = WhisperModel(
                self.model_size, device="cuda", compute_type=self.compute_type
            )
            self.device_used = f"cuda ({self.compute_type})"
        except Exception as error:
            self.fallback_reason = str(error)
            return self._load_cpu()
        return self._model

    def _fall_back_to_cpu(self, error: Exception):
        self.fallback_reason = str(error)
        self._model = None
        self.device = "cpu"
        return self._load_cpu()

    def transcribe(self, audio_path, progress_callback=None, cancel_event=None) -> list[Word]:
        model = self._load()
        try:
            return self._run(model, audio_path, progress_callback, cancel_event)
        except RuntimeError as error:
            # Missing CUDA libraries often only show up once decoding starts.
            if not self.device_used.startswith("cuda"):
                raise
            model = self._fall_back_to_cpu(error)
            return self._run(model, audio_path, progress_callback, cancel_event)

    def _run(self, model, audio_path, progress_callback, cancel_event) -> list[Word]:
        segments, info = model.transcribe(
            str(audio_path),
            language=self.language,
            word_timestamps=True,
            # Skipping silence avoids text being invented over music or noise.
            vad_filter=True,
        )
        self.detected_language = f"{info.language} ({info.language_probability:.0%})"
        words = []
        for segment in segments:
            for item in segment.words or []:
                text = item.word.strip()
                if text:
                    words.append(Word(text, item.start, item.end, item.probability))
            if progress_callback is not None and info.duration > 0:
                progress_callback(min(100.0, 100.0 * segment.end / info.duration))
            if cancel_event is not None and cancel_event.is_set():
                break
        return join_split_words(words)


def join_split_words(words: list[Word]) -> list[Word]:
    """Whisper often returns a hyphenated word as two tokens, "Co" and
    "-Pilot", which would otherwise be shown as "Co -Pilot". Glue a token that
    starts with a hyphen onto the word before it."""
    joined = []
    for word in words:
        glue = word.text[0] in HYPHENS and len(word.text) > 1 and word.text[1].isalnum()
        if glue and joined:
            previous = joined[-1]
            joined[-1] = Word(
                previous.text + word.text,
                previous.start,
                word.end,
                min(previous.confidence, word.confidence),
                previous.speaker,
            )
        else:
            joined.append(word)
    return joined


class FakeASRBackend(ASRBackend):
    """Returns canned words. Used by tests and for trying the UI without a model."""

    device_used = "fake"

    def __init__(self, words: list[Word] | None = None, delay: float = 0.0, chunks: int = 10):
        if words is None:
            words = sample_words()
        self.words = words
        self.delay = delay
        self.chunks = chunks

    def is_available(self) -> bool:
        return True

    def transcribe(self, audio_path, progress_callback=None, cancel_event=None) -> list[Word]:
        result = []
        size = max(1, -(-len(self.words) // self.chunks))
        for start in range(0, len(self.words), size):
            if cancel_event is not None and cancel_event.is_set():
                return result
            time.sleep(self.delay)
            result.extend(self.words[start : start + size])
            if progress_callback is not None:
                progress_callback(100.0 * len(result) / len(self.words))
        return result


def sample_words() -> list[Word]:
    # (sentence, seconds per word). The last one is spoken too fast to read,
    # so the fake run also produces reading-speed warnings.
    sentences = [
        ("Good evening, and welcome to the programme.", 0.35),
        ("Tonight we are looking at how the city has changed over the last twenty years,", 0.35),
        ("from its transport network to its housing.", 0.35),
        ("But first, a quick look at the headlines from around the country this afternoon.", 0.35),
        ("Unemployment figures published yesterday afternoon surprised economists.", 0.2),
    ]
    words = []
    clock = 0.5
    for sentence, pace in sentences:
        for token in sentence.split():
            words.append(Word(token, round(clock, 3), round(clock + pace - 0.05, 3)))
            clock += pace
        clock += 0.8
    return words
