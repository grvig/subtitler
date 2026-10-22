"""Entry point for the packaged Windows build."""

import os
import sys

# A windowed build has no console, so sys.stdout and sys.stderr are None, and
# libraries that print progress (the model download does) would crash.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

from subtitler.app import main  # noqa: E402

sys.exit(main())
