"""Video player with transport controls and the active subtitle shown below."""

from bisect import bisect_right

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from subtitler.core.models import Cue
from subtitler.io.srt import format_timestamp


class VideoPane(QWidget):
    position_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.video = QVideoWidget(self)
        self.player.setVideoOutput(self.video)
        self.cues = []
        self._starts = []

        # QVideoWidget draws into a native surface on Windows, and widgets
        # laid over it are often hidden, so the subtitle sits just below it.
        self.subtitle = QLabel(self)
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle.setWordWrap(True)
        self.subtitle.setMinimumHeight(56)
        self.subtitle.setStyleSheet(
            "background: black; color: white; font-size: 17px; padding: 4px;"
        )

        self.play_button = QPushButton(self)
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_button.clicked.connect(self.toggle_play)
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.sliderMoved.connect(lambda value: self.player.setPosition(value))
        self.time_label = QLabel("00:00:00.000 / 00:00:00.000", self)

        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.time_label)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.video, 1)
        layout.addWidget(self.subtitle)
        layout.addLayout(controls)

        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(self._on_duration)
        self.player.playbackStateChanged.connect(self._on_state)

    def load(self, path: str) -> None:
        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.pause()

    def toggle_play(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def seek(self, seconds: float) -> None:
        self.player.setPosition(int(max(0.0, seconds) * 1000))

    def position(self) -> float:
        return self.player.position() / 1000

    def set_cues(self, cues: list[Cue]) -> None:
        self.cues = cues
        self._starts = [cue.start for cue in cues]
        self._show_subtitle(self.position())

    def _show_subtitle(self, seconds: float) -> None:
        position = bisect_right(self._starts, seconds) - 1
        text = ""
        if position >= 0 and seconds < self.cues[position].end:
            text = "\n".join(self.cues[position].lines)
        self.subtitle.setText(text)

    def _on_duration(self, ms: int) -> None:
        self.slider.setRange(0, ms)
        self._update_time_label()

    def _update_time_label(self) -> None:
        now = format_timestamp(self.position(), ".")
        total = format_timestamp(self.player.duration() / 1000, ".")
        self.time_label.setText(f"{now} / {total}")

    def _on_position(self, ms: int) -> None:
        if not self.slider.isSliderDown():
            self.slider.setValue(ms)
        seconds = ms / 1000
        self._update_time_label()
        self._show_subtitle(seconds)
        self.position_changed.emit(seconds)

    def _on_state(self, state) -> None:
        icon = QStyle.StandardPixmap.SP_MediaPlay
        if state == QMediaPlayer.PlaybackState.PlayingState:
            icon = QStyle.StandardPixmap.SP_MediaPause
        self.play_button.setIcon(self.style().standardIcon(icon))
