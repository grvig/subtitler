"""Transcribe every media file in a folder, without the GUI.

    python -m subtitler.batch FOLDER [--profile english] [--model small]
                              [--vad on|lenient|off] [--format srt vtt]
                              [--out DIR] [--shots] [--speakers]
                              [--recursive] [--overwrite] [--fake-asr]

Subtitles are written beside each media file (or into --out) with the same
name. Files that already have subtitles are skipped unless --overwrite is
given. The exit code is 1 if any file failed.
"""

import argparse
import sys

from subtitler.core.rules.profiles import PROFILES
from subtitler.pipeline.asr import VAD_SETTINGS, FakeASRBackend, FasterWhisperBackend
from subtitler.pipeline.audio import ffmpeg_available
from subtitler.pipeline.run import WRITERS, batch_transcribe, find_media


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe a folder of media files")
    parser.add_argument("folder")
    parser.add_argument("--profile", choices=list(PROFILES), default="english")
    parser.add_argument("--model", choices=["small", "medium"], default="small")
    parser.add_argument("--vad", choices=list(VAD_SETTINGS), default="on", help="speech filter")
    parser.add_argument("--format", nargs="+", choices=list(WRITERS), default=["srt"])
    parser.add_argument("--out", help="folder for the subtitle files")
    parser.add_argument("--shots", action="store_true", help="detect shot changes too")
    parser.add_argument("--speakers", action="store_true", help="label speakers too")
    parser.add_argument("--recursive", action="store_true", help="include subfolders")
    parser.add_argument("--overwrite", action="store_true", help="redo files already done")
    parser.add_argument("--fake-asr", action="store_true", help="canned words, for testing")
    args = parser.parse_args(argv)

    if not ffmpeg_available():
        print("ffmpeg and ffprobe must be on PATH", file=sys.stderr)
        return 2
    paths = find_media(args.folder, args.recursive)
    if not paths:
        print(f"No media files in {args.folder}", file=sys.stderr)
        return 2
    if args.fake_asr:
        backend = FakeASRBackend()
    else:
        backend = FasterWhisperBackend(model_size=args.model, vad=args.vad)
    print(f"{len(paths)} files, profile {args.profile}, model {args.model}")

    items = batch_transcribe(
        paths,
        backend,
        PROFILES[args.profile],
        out_dir=args.out,
        formats=tuple(args.format),
        find_shots=args.shots,
        find_speakers=args.speakers,
        overwrite=args.overwrite,
        status=lambda message: print(message, flush=True),
    )
    failures = 0
    for item in items:
        if item.error is not None:
            failures += 1
            print(f"  FAILED   {item.media_path.name}: {item.error}")
        elif item.skipped:
            print(f"  skipped  {item.media_path.name} (subtitles already exist)")
        else:
            print(
                f"  done     {item.media_path.name}: {item.cue_count} cues, "
                f"{item.violation_count} warnings"
            )
    print(f"Device: {backend.device_used}")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
