"""Speaker labelling (diarisation) without an account or a token.

The words are grouped into speech turns at pauses, each turn is turned into
a voice embedding by the WeSpeaker ResNet34 model (VoxCeleb, CC BY 4.0, run
with onnxruntime on the CPU), and turns whose embeddings are close are
clustered into the same speaker. Every word then gets its turn's label.

A speaker change inside a turn with no pause is missed, so overlapping or
very fast exchanges come out less accurately than clear turn-taking.
"""

import threading
import wave
from collections.abc import Callable

import numpy as np

from subtitler.core.models import Word

MODEL_REPO = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
MODEL_FILE = "voxceleb_resnet34_LM.onnx"
SAMPLE_RATE = 16000

# Chosen by hand for this project and checked on a few recordings; they are
# not tuned on a labelled data set.
TURN_MAX_PAUSE = 0.4
TURN_MAX_LENGTH = 3.0
TURN_MIN_LENGTH = 1.0
SAME_SPEAKER_SIMILARITY = 0.3
MIN_SPEAKER_SECONDS = 4.0


def fbank(samples: np.ndarray) -> np.ndarray:
    """Kaldi-style log mel filterbank, 80 bins, 25 ms frames every 10 ms, with
    the per-utterance mean removed: the features the model was trained on.
    samples are floats in -1..1 at 16 kHz."""
    signal = samples.astype(np.float64) * 32768.0
    frame_length = 400
    frame_shift = 160
    fft_size = 512
    if signal.size < frame_length:
        signal = np.pad(signal, (0, frame_length - signal.size))
    count = 1 + (signal.size - frame_length) // frame_shift
    starts = np.arange(count) * frame_shift
    frames = signal[starts[:, None] + np.arange(frame_length)]
    frames = frames - frames.mean(axis=1, keepdims=True)
    shifted = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
    frames = frames - 0.97 * shifted
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame_length) / (frame_length - 1))) ** 0.85
    power = np.abs(np.fft.rfft(frames * window, fft_size)) ** 2
    features = np.log(np.maximum(power @ _mel_banks(fft_size).T, np.finfo(np.float32).eps))
    features = features - features.mean(axis=0, keepdims=True)
    return features.astype(np.float32)


def _mel_banks(fft_size: int, bins: int = 80, low: float = 20.0) -> np.ndarray:
    def mel(frequency):
        return 1127.0 * np.log(1.0 + frequency / 700.0)

    high = SAMPLE_RATE / 2
    edges = np.linspace(mel(low), mel(high), bins + 2)
    centres = mel(np.arange(fft_size // 2) * SAMPLE_RATE / fft_size)
    banks = np.zeros((bins, fft_size // 2 + 1))
    for b in range(bins):
        left, centre, right = edges[b], edges[b + 1], edges[b + 2]
        rising = (centres - left) / (centre - left)
        falling = (right - centres) / (right - centre)
        banks[b, : fft_size // 2] = np.maximum(0.0, np.minimum(rising, falling))
    return banks


class SpeakerEmbedder:
    """Loads the model on first use and turns audio into a unit-length voice
    embedding."""

    def __init__(self):
        self._session = None

    def _load(self):
        if self._session is None:
            # Imported here so that importing this module loads no model code.
            import onnxruntime
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(MODEL_REPO, MODEL_FILE)
            self._session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
        return self._session

    def embed(self, samples: np.ndarray) -> np.ndarray:
        features = fbank(samples)[None, :, :]
        vector = self._load().run(None, {"feats": features})[0][0]
        return vector / max(float(np.linalg.norm(vector)), 1e-9)


def speech_turns(words: list[Word]) -> list[list[int]]:
    """Group word positions into turns, splitting at pauses and long stretches."""
    turns = []
    for position, word in enumerate(words):
        if turns:
            previous = words[turns[-1][-1]]
            first = words[turns[-1][0]]
            pause = word.start - previous.end
            if pause <= TURN_MAX_PAUSE and word.end - first.start <= TURN_MAX_LENGTH:
                turns[-1].append(position)
                continue
        turns.append([position])
    return turns


def cluster(embeddings: np.ndarray, threshold: float = SAME_SPEAKER_SIMILARITY) -> list[int]:
    """Average-linkage agglomerative clustering on cosine similarity. Merging
    stops when no two clusters are more similar than threshold. Labels are
    numbered in order of first appearance."""
    count = len(embeddings)
    if count == 0:
        return []
    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -np.inf)
    sizes = np.ones(count)
    members = {k: [k] for k in range(count)}
    while len(members) > 1:
        flat = int(np.argmax(similarity))
        a, b = divmod(flat, count)
        if similarity[a, b] < threshold:
            break
        # Average linkage: the merged row is the size-weighted mean of both rows.
        merged = (similarity[a] * sizes[a] + similarity[b] * sizes[b]) / (sizes[a] + sizes[b])
        similarity[a, :] = merged
        similarity[:, a] = merged
        similarity[a, a] = -np.inf
        similarity[b, :] = -np.inf
        similarity[:, b] = -np.inf
        sizes[a] += sizes[b]
        members[a].extend(members.pop(b))
    labels = [0] * count
    order = sorted(members.values(), key=min)
    for number, group in enumerate(order):
        for k in group:
            labels[k] = number
    return labels


def absorb_small_speakers(
    labels: list[int], embeddings: np.ndarray, turns: list[list[int]], words: list[Word]
) -> list[int]:
    """Move the turns of any speaker with less than MIN_SPEAKER_SECONDS of
    speech to the most similar remaining speaker. A one-word fragment is too
    little audio to stand as a speaker of its own. Labels are renumbered in
    order of first appearance."""
    seconds = {}
    for turn, label in zip(turns, labels, strict=True):
        length = words[turn[-1]].end - words[turn[0]].start
        seconds[label] = seconds.get(label, 0.0) + length
    keep = [label for label in seconds if seconds[label] >= MIN_SPEAKER_SECONDS]
    if not keep:
        return [0] * len(labels)
    centres = {}
    for label in keep:
        rows = embeddings[[k for k, value in enumerate(labels) if value == label]]
        centre = rows.mean(axis=0)
        centres[label] = centre / max(float(np.linalg.norm(centre)), 1e-9)
    moved = []
    for k, label in enumerate(labels):
        if label in centres:
            moved.append(label)
        else:
            moved.append(max(keep, key=lambda kept: float(embeddings[k] @ centres[kept])))
    order = {}
    for label in moved:
        order.setdefault(label, len(order))
    return [order[label] for label in moved]


def label_speakers(
    words: list[Word],
    wav_path,
    embedder: SpeakerEmbedder,
    progress: Callable[[int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> list[str]:
    """A speaker label for each word, such as "Speaker 1"."""
    if not words:
        return []
    with wave.open(str(wav_path), "rb") as handle:
        audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0
    turns = speech_turns(words)
    vectors = []
    for number, turn in enumerate(turns):
        if cancel_event is not None and cancel_event.is_set():
            return []
        start = words[turn[0]].start
        end = words[turn[-1]].end
        # Short turns are widened around their middle, as the model needs about a second.
        if end - start < TURN_MIN_LENGTH:
            middle = (start + end) / 2
            start = middle - TURN_MIN_LENGTH / 2
            end = middle + TURN_MIN_LENGTH / 2
        first = max(0, int(start * SAMPLE_RATE))
        last = min(audio.size, int(end * SAMPLE_RATE))
        vectors.append(embedder.embed(audio[first:last]))
        if progress is not None:
            progress(int(100 * (number + 1) / len(turns)))
    embeddings = np.array(vectors)
    labels = absorb_small_speakers(cluster(embeddings), embeddings, turns, words)
    result = [""] * len(words)
    for turn, label in zip(turns, labels, strict=True):
        for position in turn:
            result[position] = f"Speaker {label + 1}"
    return result
