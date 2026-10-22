import subprocess

import pytest

from subtitler import batch
from subtitler.core.rules.profiles import ENGLISH
from subtitler.io.srt import read_srt
from subtitler.pipeline import run
from subtitler.pipeline.asr import FakeASRBackend
from subtitler.pipeline.audio import FFmpegError, ffmpeg_available
from tests.test_pipeline import write_tone


@pytest.fixture
def folder(tmp_path, monkeypatch):
    def fake_extract(media_path, out_path, *window):
        if "broken" in str(media_path):
            raise FFmpegError("cannot decode")
        write_tone(out_path, seconds=20.0)

    monkeypatch.setattr(run, "extract_audio", fake_extract)
    for name in ["a.mp4", "b.mkv", "broken.mp4", "done.mp4", "notes.txt"]:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "done.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nold\n", encoding="utf-8")
    return tmp_path


def test_find_media_ignores_other_files(folder):
    names = [path.name for path in run.find_media(folder)]
    assert names == ["a.mp4", "b.mkv", "broken.mp4", "done.mp4"]


def test_batch_writes_skips_and_reports_failures(folder):
    progress = []
    items = run.batch_transcribe(
        run.find_media(folder), FakeASRBackend(), ENGLISH, progress=progress.append
    )
    by_name = {item.media_path.name: item for item in items}
    assert by_name["broken.mp4"].error == "FFmpegError: cannot decode"
    assert by_name["done.mp4"].skipped
    assert (folder / "done.srt").read_text(encoding="utf-8").endswith("old\n")
    assert by_name["a.mp4"].cue_count > 0
    assert by_name["b.mkv"].cue_count > 0
    assert read_srt(folder / "a.srt")
    assert read_srt(folder / "b.srt")
    assert progress[-1] == 100
    assert progress == sorted(progress)


def test_batch_can_write_vtt_to_another_folder(folder, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    paths = [folder / "a.mp4"]
    run.batch_transcribe(paths, FakeASRBackend(), ENGLISH, out_dir=out, formats=("srt", "vtt"))
    assert (out / "a.srt").exists()
    assert (out / "a.vtt").read_text(encoding="utf-8").startswith("WEBVTT")


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg is not installed")
def test_command_line_tool(tmp_path, capsys):
    clip = tmp_path / "talk.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=3"]
        + ["-f", "lavfi", "-i", "color=c=gray:s=160x120:d=3:r=24", "-shortest", str(clip)],
        check=True,
    )
    assert batch.main([str(tmp_path), "--fake-asr", "--shots"]) == 0
    assert read_srt(tmp_path / "talk.srt")
    output = capsys.readouterr().out
    assert "done     talk.mp4" in output
    # A second run finds the subtitles already there.
    assert batch.main([str(tmp_path), "--fake-asr"]) == 0
    assert "skipped  talk.mp4" in capsys.readouterr().out
