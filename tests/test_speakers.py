import wave

import numpy as np
import pytest

from subtitler.core.models import Cue, Project, Word
from subtitler.core.rules.base import RuleContext
from subtitler.core.rules.layout import SpeakerChangeRule
from subtitler.core.rules.profiles import ENGLISH
from subtitler.core.segmentation import segment
from subtitler.pipeline import run, speakers
from subtitler.ui.commands import CommandStack, LabelSpeakersCommand
from tests.test_segmentation import make_words


class PitchEmbedder:
    """Stands in for the voice model: the embedding points one way for low
    tones and another way for high ones."""

    def embed(self, samples):
        spectrum = np.abs(np.fft.rfft(samples))
        peak = np.argmax(spectrum) * speakers.SAMPLE_RATE / samples.size
        if peak < 600:
            return np.array([1.0, 0.0])
        return np.array([0.0, 1.0])


def write_voices(path, pitches, seconds_each=9.0):
    rate = speakers.SAMPLE_RATE
    parts = []
    for pitch in pitches:
        t = np.arange(int(seconds_each * rate)) / rate
        parts.append(0.4 * np.sin(2 * np.pi * pitch * t))
    samples = (np.concatenate(parts) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


def test_fbank_shape_and_mean():
    features = speakers.fbank(np.zeros(16000, dtype=np.float32) + 0.01)
    assert features.shape == (98, 80)
    assert np.allclose(features.mean(axis=0), 0.0, atol=1e-4)


def test_speech_turns_split_at_pauses_and_length():
    words = make_words("one two three", word_length=0.3, gap=0.05)
    words += make_words("four five", word_length=0.3, start=words[-1].end + 1.0)
    assert speakers.speech_turns(words) == [[0, 1, 2], [3, 4]]
    long_run = make_words(" ".join(["word"] * 12), word_length=0.4, gap=0.05)
    turns = speakers.speech_turns(long_run)
    assert len(turns) > 1
    for turn in turns:
        assert long_run[turn[-1]].end - long_run[turn[0]].start <= speakers.TURN_MAX_LENGTH


def test_cluster_two_groups_labelled_in_order():
    rng = np.random.default_rng(1)
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    rows = [b + 0.05 * rng.standard_normal(3) for _ in range(3)]
    rows += [a + 0.05 * rng.standard_normal(3) for _ in range(4)]
    embeddings = np.array([row / np.linalg.norm(row) for row in rows])
    assert speakers.cluster(embeddings) == [0, 0, 0, 1, 1, 1, 1]
    assert speakers.cluster(np.zeros((0, 3))) == []


def test_small_speakers_are_absorbed():
    words = make_words(" ".join(["word"] * 10), word_length=0.5, gap=0.6)
    turns = [[k] for k in range(10)]
    embeddings = np.array([[1.0, 0.0]] * 9 + [[0.9, 0.44]])
    labels = speakers.absorb_small_speakers([0] * 9 + [1], embeddings, turns, words)
    assert labels == [0] * 10


def test_label_speakers_separates_two_voices(tmp_path):
    wav = tmp_path / "voices.wav"
    write_voices(wav, [300, 900])
    # Ten words a voice, so each speaker has more than MIN_SPEAKER_SECONDS of speech.
    words = make_words(" ".join(["low"] * 10), word_length=0.5, gap=0.3)
    words += make_words(" ".join(["high"] * 10), word_length=0.5, gap=0.3, start=9.2)
    labels = speakers.label_speakers(words, wav, PitchEmbedder())
    assert labels == ["Speaker 1"] * 10 + ["Speaker 2"] * 10


def test_too_little_speech_is_one_speaker(tmp_path):
    wav = tmp_path / "voices.wav"
    write_voices(wav, [300, 900])
    words = make_words("low voice", word_length=0.5, gap=0.3)
    words += make_words("high voice", word_length=0.5, gap=0.3, start=9.2)
    labels = speakers.label_speakers(words, wav, PitchEmbedder())
    assert set(labels) == {"Speaker 1"}


def test_segmentation_splits_at_a_speaker_change():
    words = make_words("we could meet at the station tomorrow morning around nine")
    for position, word in enumerate(words):
        word.speaker = "Speaker 1"
        if position >= 6:
            word.speaker = "Speaker 2"
    cues = segment(words, [], ENGLISH)
    assert len(cues) == 2
    assert cues[1].words[0].text == "tomorrow"


def test_speaker_change_rule():
    rule = SpeakerChangeRule()
    context = RuleContext(profile=ENGLISH)
    one = Cue(1, 0.0, 2.0, ["Hi"], [Word("Hi", 0.0, 1.0, speaker="Speaker 1")])
    mixed = [
        Word("Hi", 0.0, 0.5, speaker="Speaker 1"),
        Word("there", 0.6, 1.0, speaker="Speaker 2"),
    ]
    two = Cue(1, 0.0, 2.0, ["Hi there"], mixed)
    assert rule.check(one, context) == []
    assert len(rule.check(two, context)) == 1


def test_label_speakers_command_undo():
    words = make_words("a b c")
    project = Project(cues=[Cue(1, 0.0, 2.0, ["a b c"], words)])
    stack = CommandStack()
    stack.execute(LabelSpeakersCommand(project, ["Speaker 1", "Speaker 1", "Speaker 2"]))
    assert [w.speaker for w in words] == ["Speaker 1", "Speaker 1", "Speaker 2"]
    stack.undo()
    assert [w.speaker for w in words] == [None, None, None]
    with pytest.raises(ValueError):
        LabelSpeakersCommand(project, ["Speaker 1"])


def test_transcription_can_label_speakers(tmp_path, monkeypatch):
    def fake_extract(media_path, out_path, *window):
        write_voices(out_path, [300, 900])

    monkeypatch.setattr(run, "extract_audio", fake_extract)

    class TwoVoiceASR:
        device_used = "fake"

        def transcribe(self, audio_path, progress_callback=None, cancel_event=None, prompt=None):
            first = make_words("the low voice is talking. " * 2, word_length=0.5, gap=0.3)
            second = make_words(
                "the high voice answers it. " * 2, word_length=0.5, gap=0.3, start=9.2
            )
            return first + second

    result = run.transcribe_media(
        "clip.mp4", TwoVoiceASR(), ENGLISH, find_speakers=True, embedder=PitchEmbedder()
    )
    assert {w.speaker for w in result.words} == {"Speaker 1", "Speaker 2"}
    assert all(len({w.speaker for w in cue.words}) == 1 for cue in result.cues)
