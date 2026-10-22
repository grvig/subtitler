# PyInstaller build description. Run it through packaging/build.py, which
# sets SUBTITLER_WITH_CUDA and SUBTITLER_WITH_FFMPEG from its options.

import os
import shutil
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH).parent
WITH_CUDA = os.environ.get("SUBTITLER_WITH_CUDA") == "1"
WITH_FFMPEG = os.environ.get("SUBTITLER_WITH_FFMPEG") == "1"

# faster-whisper ships the voice activity model as data, and uniseg reads
# its Unicode tables from files.
datas = collect_data_files("faster_whisper") + collect_data_files("uniseg")
binaries = collect_dynamic_libs("ctranslate2")

if WITH_CUDA:
    import importlib.util

    spec = importlib.util.find_spec("nvidia")
    if spec is None:
        raise SystemExit("--with-cuda needs nvidia-cublas-cu12 and nvidia-cudnn-cu12 installed")
    for root in spec.submodule_search_locations:
        for dll in Path(root).glob("*/bin/*.dll"):
            binaries.append((str(dll), f"nvidia/{dll.parent.parent.name}/bin"))

if WITH_FFMPEG:
    for tool in ["ffmpeg", "ffprobe"]:
        found = shutil.which(tool)
        if found is None:
            raise SystemExit(f"--with-ffmpeg needs {tool} on PATH")
        # winget puts a link on PATH; bundle the real program behind it.
        binaries.append((str(Path(found).resolve()), "ffmpeg"))

analysis = Analysis(
    [str(ROOT / "packaging" / "launch.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=["scenedetect.backends.opencv"],
    # Installed alongside in many environments and picked up through optional
    # imports, but never used by the app.
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "pytest",
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "tensorflow",
        "scipy",
        "pandas",
        "pyarrow",
        "sympy",
    ],
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="Subtitler",
    console=False,
)
COLLECT(exe, analysis.binaries, analysis.datas, name="Subtitler")
