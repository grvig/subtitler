"""Build the Windows app into dist/Subtitler/Subtitler.exe.

    python packaging/build.py [--with-cuda] [--with-ffmpeg] [--out FOLDER]

--with-cuda    include the cuBLAS and cuDNN DLLs from the nvidia-cublas-cu12
               and nvidia-cudnn-cu12 packages, so the GPU works without a
               CUDA install (adds about 1 GB)
--with-ffmpeg  include the ffmpeg and ffprobe found on PATH, so the app does
               not need ffmpeg installed
--out          where to put the dist and build folders (default: the
               repository); worth pointing outside a synced folder such as
               OneDrive, as the build is large

The speech model is not included. It is downloaded on first use, as when
running from source.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Windows app")
    parser.add_argument("--with-cuda", action="store_true")
    parser.add_argument("--with-ffmpeg", action="store_true")
    parser.add_argument("--out", default=str(ROOT))
    args = parser.parse_args()
    environment = dict(os.environ)
    environment["SUBTITLER_WITH_CUDA"] = str(int(args.with_cuda))
    environment["SUBTITLER_WITH_FFMPEG"] = str(int(args.with_ffmpeg))
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(ROOT / "packaging" / "subtitler.spec"),
        "--noconfirm",
        "--distpath",
        str(Path(args.out) / "dist"),
        "--workpath",
        str(Path(args.out) / "build"),
    ]
    return subprocess.run(command, env=environment, cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
