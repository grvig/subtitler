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
after another.

Text is measured in grapheme clusters, not codepoints, and line breaks come
from the Unicode line breaking rules rather than splitting on spaces, so
Tamil and Devanagari text is counted and broken correctly.

## Status

This is the first part of a semester project. Working now:

- transcription with faster-whisper, on the GPU with a CPU fallback
- segmentation into cues, and a rule checker that lists problems
- editor with video, waveform, cue table and problem list
- undo and redo for text edits, timing changes, split, merge and re-segment
- SRT import and export, WebVTT export, project files with autosave

Not done yet: shot change detection (the segmenter already accepts a list of
cut times, it is just always empty for now), re-aligning edited text to the
audio, speaker labels, batch processing and packaging.

**The numbers are placeholders.** The reading speed, line length and duration
limits in `subtitler/core/rules/profiles.py` and the cost weights were picked
by hand so the program could be built and tested. They are not taken from any
broadcaster's style guide yet and need to be replaced with published values.

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

## Using the editor

| Key | Action |
| --- | --- |
| Space | play / pause |
| I / O | set the selected cue's start / end to the playhead |
| Ctrl+Return | split the cue at the playhead |
| Ctrl+M | merge the selected cue with the next one |
| Ctrl+Z / Ctrl+Y | undo / redo |
| Ctrl+R | re-segment |
| Ctrl+E | export SRT |

- Double-click a cue's text to edit it. Separate two lines with `|`.
- An edited cue is locked (shown in italics). Re-segmenting rebuilds every
  other cue but leaves locked ones exactly as they are.
- Drag the edge of a cue in the waveform to change its timing. Click
  elsewhere in the waveform to seek, and scroll to zoom.
- Double-click a problem in the list on the right to jump to that cue.

## Project layout

```
subtitler/
  core/       models, text measurement, line breaking, segmentation, rules
  io/         SRT, WebVTT and project files
  pipeline/   audio extraction, speech recognition, background jobs
  ui/         Qt widgets and undoable commands
  app.py      entry point
tests/
```

`core` does not import Qt or any machine learning library, so it can be
tested with nothing but pytest.

## Tests

```bash
python -m pytest
```

The tests use a fake speech recogniser and generated audio. They need no model
download, no media file and no ffmpeg.

Linting and formatting use ruff:

```bash
ruff check .
ruff format .
```

`pre-commit install` sets both up to run, together with the tests, before
each commit.
