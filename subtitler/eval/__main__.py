"""Evaluate the segmenter against reference subtitles.

    python -m subtitler.eval FOLDER [--profile english] [--model small]
                             [--language en] [--sweep 200] [--report FILE]

FOLDER holds media files, each with a reference subtitle file of the same
name and the extension .srt (scripts/fetch_eval_media.py fills one). For
each film the speech recogniser and shot detector run once, and their
output is cached beside the media as .words.json and .shots.json, so later
runs only redo the segmentation.

Three systems are scored: the reference itself (compliance only), the
greedy baseline in eval/baseline.py, and the optimiser in core/segmentation.

With --sweep N, N random sets of cost weights are tried. Tuning is done
two-fold: weights are chosen on one film and scored on the other, so the
reported gain is never measured on the film it was tuned on.
"""

import argparse
import json
import random
import sys
import tempfile
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

from subtitler.core.models import Word
from subtitler.core.rules.profiles import PROFILES, default_cost_weights
from subtitler.core.segmentation import segment
from subtitler.eval.baseline import greedy_segment
from subtitler.eval.metrics import boundary_scores, compliance
from subtitler.io.srt import read_srt
from subtitler.pipeline.asr import VAD_SETTINGS, FasterWhisperBackend
from subtitler.pipeline.audio import extract_audio
from subtitler.pipeline.run import find_media
from subtitler.pipeline.shots import detect_shots

# Score used to rank weight sets: boundary F1, less the share of cues with an
# error-level violation. Chosen for this project.
ERROR_WEIGHT = 1.0
SEED = 7


def load_film(media: Path, backend_factory, language, vad: str) -> dict:
    words_cache = media.with_suffix(f".words-{vad}.json")
    shots_cache = media.with_suffix(".shots.json")
    if not words_cache.exists():
        print(f"  transcribing {media.name}", flush=True)
        backend = backend_factory()
        backend.language = language
        with tempfile.TemporaryDirectory() as work_dir:
            wav = Path(work_dir) / "audio.wav"
            extract_audio(media, wav)
            words = backend.transcribe(wav)
        words_cache.write_text(json.dumps([asdict(w) for w in words]), encoding="utf-8")
        print(f"    {len(words)} words on {backend.device_used}", flush=True)
    if not shots_cache.exists():
        print(f"  detecting shots in {media.name}", flush=True)
        shots_cache.write_text(json.dumps(detect_shots(media)), encoding="utf-8")
    words = [Word(**item) for item in json.loads(words_cache.read_text(encoding="utf-8"))]
    shots = json.loads(shots_cache.read_text(encoding="utf-8"))
    reference = read_srt(media.with_suffix(".srt"))
    return {"name": media.stem, "words": words, "shots": shots, "reference": reference}


def score(cues, film, profile) -> dict:
    result = compliance(cues, profile, film["shots"])
    result.update(boundary_scores(cues, film["reference"]))
    result["objective"] = result["f1"] - ERROR_WEIGHT * result.get("error_rate", 0.0)
    return result


def optimise(film, profile, weights) -> dict:
    tuned = replace(profile, cost_weights=weights)
    return score(segment(film["words"], film["shots"], tuned), film, profile)


def random_weights(rng: random.Random) -> dict:
    # Each weight is scaled by a factor between 0.1 and 10 of its default.
    weights = {}
    for name, value in default_cost_weights().items():
        weights[name] = value * 10 ** rng.uniform(-1, 1)
    return weights


def sweep(films, profile, trials: int) -> list[dict]:
    rng = random.Random(SEED)
    candidates = [default_cost_weights()] + [random_weights(rng) for _ in range(trials)]
    folds = []
    for held_out in films:
        tuning = [film for film in films if film is not held_out]
        if not tuning:
            tuning = [held_out]
        best = None
        best_score = None
        for weights in candidates:
            value = sum(optimise(f, profile, weights)["objective"] for f in tuning) / len(tuning)
            if best_score is None or value > best_score:
                best = weights
                best_score = value
        folds.append(
            {
                "test": held_out["name"],
                "weights": best,
                "default": optimise(held_out, profile, default_cost_weights()),
                "tuned": optimise(held_out, profile, best),
            }
        )
    return folds


COLUMNS = [
    ("cues", "cues", "{:.0f}"),
    ("f1", "boundary F1", "{:.2f}"),
    ("precision", "precision", "{:.2f}"),
    ("recall", "recall", "{:.2f}"),
    ("start_error", "start err s", "{:.2f}"),
    ("error_rate", "cues w/ errors", "{:.0%}"),
    ("rate_reading-speed", "too fast", "{:.0%}"),
    ("rate_break-quality", "bad breaks", "{:.0%}"),
    ("rate_shot-crossing", "cross cuts", "{:.0%}"),
    ("mean_cps", "mean cps", "{:.1f}"),
]


def table(rows: list[tuple[str, dict]]) -> list[str]:
    header = "| system | " + " | ".join(title for _key, title, _fmt in COLUMNS) + " |"
    lines = [header, "|" + "---|" * (len(COLUMNS) + 1)]
    for label, values in rows:
        cells = []
        for key, _title, fmt in COLUMNS:
            if key in values:
                cells.append(fmt.format(values[key]))
            elif key.startswith("rate_") or key == "error_rate":
                cells.append(fmt.format(0.0))
            else:
                cells.append("-")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate subtitle segmentation")
    parser.add_argument("folder")
    parser.add_argument("--profile", choices=list(PROFILES), default="english")
    parser.add_argument("--model", choices=["small", "medium"], default="small")
    parser.add_argument("--language", default=None, help="skip language detection, e.g. en")
    parser.add_argument("--vad", choices=list(VAD_SETTINGS), default="on")
    parser.add_argument("--sweep", type=int, default=0, help="random weight sets to try")
    parser.add_argument("--report", help="write the results as Markdown to this file")
    args = parser.parse_args(argv)
    profile = PROFILES[args.profile]

    media = [p for p in find_media(args.folder) if p.with_suffix(".srt").exists()]
    if not media:
        print("No media files with a matching .srt in that folder", file=sys.stderr)
        return 2
    print(f"Loading {len(media)} films")
    films = []
    for path in media:
        backend_factory = partial(FasterWhisperBackend, args.model, vad=args.vad)
        films.append(load_film(path, backend_factory, args.language, args.vad))

    report = [f"# Segmentation evaluation ({profile.name} profile, filter {args.vad})", ""]
    for film in films:
        rows = [
            ("reference", compliance(film["reference"], profile, film["shots"])),
            ("greedy baseline", score(greedy_segment(film["words"], profile), film, profile)),
            ("optimiser", optimise(film, profile, default_cost_weights())),
        ]
        report += [f"## {film['name']}", ""]
        report += [
            f"{len(film['words'])} recognised words, {len(film['shots'])} shot changes, "
            f"{len(film['reference'])} reference cues.",
            "",
        ]
        report += table(rows) + [""]

    if args.sweep:
        print(f"Sweeping {args.sweep} weight sets", flush=True)
        report += [f"## Weight tuning ({args.sweep} random sets, two-fold)", ""]
        for fold in sweep(films, profile, args.sweep):
            report += [f"### Tested on {fold['test']}", ""]
            report += table([("default weights", fold["default"]), ("tuned", fold["tuned"])])
            chosen = ", ".join(f"{k} {v:.2f}" for k, v in fold["weights"].items())
            report += ["", f"Tuned weights: {chosen}", ""]

    text = "\n".join(report)
    print(text)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
