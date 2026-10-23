# Subtitler

A desktop app that turns a local video or audio file into a subtitle file,
with a person checking and correcting the result. After the speech model has
been downloaded once, everything runs offline.

Speech recognition is treated as an input, not the point of the project. The
work is in turning a stream of timed words into subtitles that are actually
readable:

- reading speed (characters per second) stays under a limit
- cues stay on screen long enough to read, but not much longer than the speech
- no more than two lines per cue, each under a length limit
- lines break at sensible places, not after "the" or inside "well-known"
- cues avoid running across sentence ends, pauses and shot changes

All of these are weighed together by one dynamic programming pass over the
words (see `subtitler/core/segmentation.py`), instead of being patched on one
after another. `docs/evaluation.md` compares it with a greedy method of the
usual kind on two open films with published subtitles.

Text is measured in grapheme clusters, not codepoints, and line breaks come
from the Unicode line breaking rules rather than splitting on spaces, so
Tamil and Devanagari text is counted and broken correctly.

## Status

Working:

- transcription with faster-whisper, on the GPU with a CPU fallback
- shot change detection with PySceneDetect; cue edges snap to the cuts
- segmentation into cues, and a rule checker that lists problems
- editor with video, waveform, cue table and problem list
- undo and redo for text edits, timing changes, split, merge, re-segment
  and realign
- realigning corrected text to the audio, so corrections survive re-segmenting
- batch transcription of a folder, from the menu or the command line
- speaker labels, with cues split where the speaker changes
- SRT import and export, WebVTT export, project files with autosave
- a packaged Windows build

### Where the limits come from

The style profiles in `subtitler/core/rules/profiles.py` (English, Hindi,
Tamil) take their limits from the Netflix Timed Text Style Guides for adult
programmes: 42 characters per line, two lines, 5/6 s to 7 s on screen, at
least two frames between subtitles, 20 characters per second for English and
22 for Hindi and Tamil, and the guide's rules for timing around shot changes.
The source of each value is written next to it.

Some numbers are this project's own and are marked as such: the reading speed
the segmenter aims for (below the maximum), and the weights of the cost
function. The guides also do not say how to count characters in Indic
scripts; this project counts grapheme clusters.

## Requirements

- Windows, macOS or Linux with Python 3.11 or newer
- **ffmpeg** and **ffprobe** on your PATH. They are used to read audio out of
  media files. On Windows: `winget install Gyan.FFmpeg`, then open a new
  terminal. If they are missing the app still starts, shows a warning and
  disables Transcribe.
- Optional: an NVIDIA GPU for faster transcription

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

(On macOS or Linux, activate with `source .venv/bin/activate`.)

For the GPU on Windows, the CUDA libraries that faster-whisper needs can be
installed with pip:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

The app finds them automatically. Without them it falls back to the CPU and
the status bar says so.

## Run

```bash
python -m subtitler.app
python -m subtitler.app path\to\video.mp4
python -m subtitler.app --fake-asr
```

`--fake-asr` uses a fixed set of words instead of the speech model, which is
handy for trying the editor without downloading anything.

The first real transcription downloads the Whisper model (about 500 MB for
"small") into the Hugging Face cache in your home folder.

### Choosing a model size

The default model is **small** with `int8_float16` weights. That was chosen
for a laptop RTX 3050 Ti with 4 GB of video memory, where it transcribes a
three minute clip in roughly 15 to 30 seconds.

**medium** can be picked from the toolbar. It is more accurate but slower and
close to the limit of a 4 GB card; if it runs out of memory, go back to small.
large-v3 does not fit in 4 GB and is not offered.

On the CPU the model runs with `int8` weights on 8 threads.

Whisper detects the spoken language by itself, and the status bar shows what
it found. It goes by the audio: a dubbed video is transcribed in the language
of its dub.

### Speech filter

Before transcribing, a voice activity filter skips audio that does not sound
like speech. **on** is the default and the safe choice for anything with music
or sound effects. **lenient** and **off** catch more quiet speech, but over
music Whisper may then write text that nobody said; see `docs/evaluation.md`.

### Batch transcription

**File > Batch Transcribe Folder** does every media file in a folder and saves
an `.srt` beside each one. The same from a terminal, with more options:

```bash
python -m subtitler.batch path\to\folder --profile english --shots --speakers --format srt vtt
```

Files that already have subtitles are skipped unless `--overwrite` is given.

## Using the editor

| Key | Action |
| --- | --- |
| Space | play / pause |
| I / O | set the selected cue's start / end to the playhead |
| Ctrl+Return | split the cue at the playhead |
| Ctrl+M | merge the selected cue with the next one |
| Ctrl+Z / Ctrl+Y | undo / redo |
| Ctrl+D | detect shot changes |
| Ctrl+L | realign edited cues |
| Ctrl+K | label speakers |
| Ctrl+R | re-segment |
| Ctrl+E | export SRT |

- Double-click a cue's text to edit it. Separate two lines with `|`.
- An edited cue is locked (shown in italics). Re-segmenting rebuilds every
  other cue but leaves locked ones exactly as they are.
- **Realign Edited Cues** listens to each edited cue's audio again and gives
  the corrected text its own word timings. The cue is then unlocked, and
  re-segmenting can regroup it without losing the correction. It also gives
  word timings to imported subtitles, so they can be re-segmented too.
- **Detect Shot Changes** marks the cuts on the waveform as dashed lines.
  Re-segment afterwards to fit the cues to them.
- **Label Speakers** works out who is speaking and shows it in the Speaker
  column. Re-segment afterwards and cues are split where the speaker
  changes; a cue that still holds two speakers is flagged. It needs no
  account or token. Speech is grouped into turns at pauses, so it works best
  on clear turn-taking such as interviews; with heavy music and effects, as in
  film soundtracks, it tends to report too many speakers.
- Drag the edge of a cue in the waveform to change its timing. Click
  elsewhere in the waveform to seek, and scroll to zoom.
- Double-click a problem in the list on the right to jump to that cue.

## Packaged Windows build

```bash
pip install pyinstaller
python packaging/build.py --with-cuda --with-ffmpeg --out C:\builds\subtitler
```

This makes `dist\Subtitler\Subtitler.exe` in the `--out` folder. With both
options the folder carries the CUDA libraries and ffmpeg, so the target
machine needs neither installed. That build is about 3 GB, two thirds of it
the CUDA libraries; without `--with-cuda` it is under 1 GB. The speech model
is not included and is downloaded on first use.

## Evaluation

```bash
python scripts/fetch_eval_media.py
python -m subtitler.eval %USERPROFILE%\subtitler-eval --language en --sweep 200
```

The first command downloads two Blender Foundation films with their published
subtitles. The second scores the reference subtitles, a greedy baseline and
the optimiser, and tries random cost weights. Results and method are in
`docs/evaluation.md`.

## Models used

- Speech recognition: OpenAI Whisper, through faster-whisper, downloaded on
  first use.
- Speaker labels: the WeSpeaker ResNet34 model trained on VoxCeleb
  ([Wespeaker/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM),
  CC BY 4.0), from the WeSpeaker toolkit by Wang et al., run with
  onnxruntime. About 27 MB, downloaded on first use.

## Project layout

```
subtitler/
  core/       models, text measurement, line breaking, segmentation,
              realignment, rules
  io/         SRT, WebVTT and project files
  pipeline/   audio extraction, speech recognition, shot detection, speaker
              labels, the processing steps (run.py) and their background jobs
  ui/         Qt widgets and undoable commands
  eval/       evaluation metrics, greedy baseline, evaluation runner
  app.py      editor entry point
  batch.py    batch command line tool
scripts/      evaluation media download
packaging/    PyInstaller build
docs/         evaluation results
tests/
```

`core` does not import Qt or any machine learning library, so it can be
tested with nothing but pytest.

## Tests

```bash
python -m pytest
```

The tests use a fake speech recogniser and generated audio, so no model is
downloaded. A few tests make short clips with ffmpeg and are skipped when it
is not installed.

Linting and formatting use ruff:

```bash
ruff check .
ruff format .
```

`pre-commit install` sets both up to run, together with the tests, before
each commit.
