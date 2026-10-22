"""Download the open films used by the evaluation, with their official
English subtitles, into a folder outside the repository.

    python scripts/fetch_eval_media.py [TARGET_FOLDER]

The default folder is ~/subtitler-eval. Files already there are kept. Both
films are Creative Commons Attribution licensed productions of the Blender
Foundation (blender.org); the subtitles are the ones published with them.

  Tears of Steel (2012), CC BY 3.0   https://mango.blender.org
  Sintel (2010), CC BY 3.0           https://durian.blender.org
"""

import shutil
import sys
import urllib.request
from pathlib import Path

# (file name, url, approximate size in MB)
FILES = [
    (
        "tears_of_steel_720p.mov",
        "https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov",
        355,
    ),
    (
        "tears_of_steel_720p.srt",
        "https://download.blender.org/demo/movies/ToS/subtitles/TOS-en.srt",
        0,
    ),
    (
        "sintel-1024-surround.mp4",
        # Mirror listed on durian.blender.org/download.
        "http://peach.themazzone.com/durian/movies/sintel-1024-surround.mp4",
        123,
    ),
    (
        "sintel-1024-surround.srt",
        "https://durian.blender.org/wp-content/content/subtitles/sintel_en.srt",
        0,
    ),
]


def main() -> int:
    target = Path.home() / "subtitler-eval"
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)
    for name, url, size in FILES:
        path = target / name
        if path.exists() and path.stat().st_size > 0:
            print(f"have     {name}")
            continue
        print(f"fetching {name} (about {size} MB) from {url}", flush=True)
        partial = path.with_suffix(path.suffix + ".part")
        # Some of the hosts refuse requests with Python's default user agent.
        request = urllib.request.Request(url, headers={"User-Agent": "subtitler-eval-fetch/1.0"})
        with urllib.request.urlopen(request) as response, open(partial, "wb") as handle:
            shutil.copyfileobj(response, handle)
        partial.replace(path)
    print(f"Files are in {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
