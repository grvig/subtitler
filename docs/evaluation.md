# Evaluation

This compares the segmenter against professionally published subtitles and
against a simple greedy method, on the same recognised words.

## Data

Two open films from the Blender Foundation, both CC BY 3.0, with the English
subtitle files published alongside them:

| film | length | reference cues | reference words |
|---|---|---|---|
| Tears of Steel (2012) | 12 min | 76 | 334 |
| Sintel (2010) | 15 min | 26 | 110 |

`scripts/fetch_eval_media.py` downloads them (about 480 MB) to
`~/subtitler-eval`, outside the repository.

This is a small set, and Sintel in particular has very little dialogue. The
numbers below show which way things point; they are not statistically
strong. The reference files are also not held to the Netflix rules this
project checks against: 30 % and 65 % of their cues cross a shot change
without the speech crossing it, and 14 % of the Tears of Steel cues are over
20 characters per second.

## Method

Whisper "small" (`int8_float16` on an RTX 3050 Ti) transcribes each film
once, with the language set to English, and PySceneDetect finds the shot
changes once. Both results are cached, so every system below segments
exactly the same words with the same cuts.

- **reference**: the published subtitle file.
- **greedy baseline** (`subtitler/eval/baseline.py`): adds words to a cue
  until the next word would break the character or duration limit, or a
  sentence has ended; fills lines first line first; times cues exactly to
  their words. This is how transcription tools usually build subtitles.
- **optimiser**: `subtitler/core/segmentation.py` with the default weights.

Measures (`subtitler/eval/metrics.py`):

- **boundary F1**: a cue start counts as matching a reference cue start
  within 0.3 s. Text is not compared, because the recogniser's wording never
  matches a reference exactly.
- **start err**: mean distance between matched starts, in seconds.
- **cues w/ errors**: share of cues with an error-level rule violation
  (reading speed over the maximum, line too long, too many lines, overlap).
- **too fast, bad breaks, cross cuts**: share of cues flagged by the
  reading-speed, break-quality and shot-crossing rules.

Run it with:

```bash
python -m subtitler.eval %USERPROFILE%\subtitler-eval --language en --sweep 200
```

## Results

Speech filter on (the default):

### Sintel

| system | cues | boundary F1 | precision | recall | start err s | cues w/ errors | too fast | bad breaks | cross cuts | mean cps |
|---|---|---|---|---|---|---|---|---|---|---|
| reference | 26 | - | - | - | - | 0% | 0% | 8% | 65% | 7.8 |
| greedy baseline | 10 | 0.28 | 0.50 | 0.19 | 0.13 | 50% | 50% | 20% | 0% | 17.9 |
| optimiser | 13 | 0.36 | 0.54 | 0.27 | 0.14 | 0% | 0% | 0% | 8% | 13.3 |

### Tears of Steel

| system | cues | boundary F1 | precision | recall | start err s | cues w/ errors | too fast | bad breaks | cross cuts | mean cps |
|---|---|---|---|---|---|---|---|---|---|---|
| reference | 76 | - | - | - | - | 14% | 14% | 3% | 30% | 13.3 |
| greedy baseline | 47 | 0.33 | 0.43 | 0.26 | 0.14 | 28% | 28% | 9% | 0% | 16.2 |
| optimiser | 58 | 0.36 | 0.41 | 0.32 | 0.13 | 3% | 3% | 0% | 3% | 11.3 |

On both films the optimiser finds more of the reference boundaries than the
greedy method, and its output breaks the rules far less often: no cue in
Sintel and 3 % in Tears of Steel are too fast to read, against 50 % and 28 %
for the greedy method, and it never ends a line on a function word.

Recall is low for every system mostly because of the recogniser, not the
segmenter: with the filter on, Whisper returns 60 of Sintel's 110 words and
260 of Tears of Steel's 334, so many reference cues have no words to match.

## Speech filter

faster-whisper's voice activity filter drops audio it does not judge to be
speech. Both films have loud music and effects under the dialogue, so the
filter was tried at three settings (`--vad`):

| filter | Sintel words | Sintel F1 greedy / optimiser | Tears of Steel words | Tears of Steel F1 greedy / optimiser |
|---|---|---|---|---|
| on (0.5) | 60 | 0.28 / 0.36 | 260 | 0.33 / 0.36 |
| lenient (0.25) | 68 | 0.32 / 0.33 | 289 | 0.39 / 0.39 |
| off | 165 | 0.03 / 0.11 | 360 | 0.40 / 0.42 |

With the filter off, Sintel collapses: Whisper writes 165 words against a
reference of 110, inventing text over the score, and the cues stop lining up
with anything. Tears of Steel, which is mostly dialogue, gains a little. The
filter therefore stays on by default and the setting is exposed in the
toolbar and the batch tool.

## Weight tuning

200 random sets of cost weights, each weight between a tenth and ten times
its default, were scored by boundary F1 less the share of cues with errors.
Tuning was two-fold: the best weights on one film were scored on the other.

| tested on | default F1 | tuned F1 | default errors | tuned errors |
|---|---|---|---|---|
| Sintel | 0.36 | 0.36 | 0% | 0% |
| Tears of Steel | 0.36 | 0.38 | 3% | 5% |

Tuning did not produce a gain that carries over between films, and the
weights chosen on each film were very different from each other. With two
films that is noise, so the default weights are kept. The sweep takes about
16 seconds, so it is cheap to repeat once more reference material is added.

## What would make this stronger

- More films, especially dialogue-heavy ones, with subtitles made to a known
  style guide.
- Measuring line-break agreement as well as cue boundaries, which needs the
  recognised words aligned to the reference text.
- Comparing against the segments Whisper itself produces, as a second
  baseline.
