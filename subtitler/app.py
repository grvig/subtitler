"""Entry point: python -m subtitler.app [media file] [--fake-asr]"""

import argparse
import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from subtitler.pipeline.audio import ffmpeg_available
from subtitler.ui.main_window import MainWindow


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline subtitle editor")
    parser.add_argument("media", nargs="?", help="video or audio file to open")
    parser.add_argument(
        "--fake-asr",
        action="store_true",
        help="use canned words instead of the speech model (for trying the UI)",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Subtitler")
    window = MainWindow(use_fake_asr=args.fake_asr)
    window.show()
    if not ffmpeg_available():
        QMessageBox.critical(
            window,
            "ffmpeg not found",
            "ffmpeg and ffprobe were not found on PATH.\n\n"
            "They are needed to read audio from media files, so Transcribe is "
            "disabled. Install ffmpeg, make sure the folder containing ffmpeg.exe "
            "is on PATH, and restart the application.\n\n"
            "Importing, editing and exporting subtitles still works.",
        )
    if args.media:
        window.open_media(args.media)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
