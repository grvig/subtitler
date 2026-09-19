"""Waveform strip showing the audio envelope, cue regions and the playhead.

Clicking seeks. Dragging the left or right edge of a cue changes its timing;
the widget only previews the drag and emits timing_changed on release, and
the main window turns that into an undoable SetTimingCommand.
"""

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from subtitler.core.models import Cue

EDGE_GRAB_PIXELS = 6
MIN_CUE_SECONDS = 0.1
DEFAULT_PIXELS_PER_SECOND = 80.0


class WaveformView(QWidget):
    seek_requested = Signal(float)
    timing_changed = Signal(int, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(110)
        self.setMaximumHeight(200)
        self.setMouseTracking(True)
        self.peaks_min = np.zeros(0, dtype=np.float32)
        self.peaks_max = np.zeros(0, dtype=np.float32)
        self.buckets_per_second = 100
        self.cues = []
        self.selected = -1
        self.playhead = 0.0
        self.view_start = 0.0
        self.pixels_per_second = DEFAULT_PIXELS_PER_SECOND
        # (cue position, "start" or "end", start, end) while an edge is dragged
        self.drag = None

    def set_peaks(self, peaks_min, peaks_max, buckets_per_second: int) -> None:
        self.peaks_min = peaks_min
        self.peaks_max = peaks_max
        self.buckets_per_second = buckets_per_second
        self.update()

    def set_cues(self, cues: list[Cue]) -> None:
        self.cues = cues
        self.update()

    def set_selected(self, position: int) -> None:
        self.selected = position
        self.update()

    def set_playhead(self, seconds: float) -> None:
        self.playhead = seconds
        visible = self.width() / self.pixels_per_second
        # Page the view along so the playhead stays on screen.
        if seconds < self.view_start or seconds > self.view_start + visible * 0.9:
            self.view_start = max(0.0, seconds - visible * 0.1)
        self.update()

    def time_at(self, x: float) -> float:
        return self.view_start + x / self.pixels_per_second

    def x_at(self, seconds: float) -> float:
        return (seconds - self.view_start) * self.pixels_per_second

    def _cue_times(self, position: int) -> tuple[float, float]:
        if self.drag is not None and self.drag[0] == position:
            return self.drag[2], self.drag[3]
        return self.cues[position].start, self.cues[position].end

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        width = self.width()
        height = self.height()
        middle = height / 2
        painter.fillRect(self.rect(), QColor(24, 26, 30))

        for position in range(len(self.cues)):
            start, end = self._cue_times(position)
            left = self.x_at(start)
            right = self.x_at(end)
            if right < 0 or left > width:
                continue
            colour = QColor(70, 140, 230, 60)
            if position == self.selected:
                colour = QColor(240, 180, 60, 90)
            painter.fillRect(QRectF(left, 0, right - left, height), colour)
            painter.setPen(QPen(colour.lighter(160), 1))
            painter.drawLine(QPointF(left, 0), QPointF(left, height))
            painter.drawLine(QPointF(right, 0), QPointF(right, height))
            painter.drawText(QRectF(left + 3, 2, max(0.0, right - left - 6), 16), str(position + 1))

        if self.peaks_max.size and width > 0:
            # Buckets covered by each pixel column, reduced to one min and max per column.
            times = self.view_start + np.arange(width + 1) / self.pixels_per_second
            edges = np.clip(
                (times * self.buckets_per_second).astype(np.int64), 0, self.peaks_max.size
            )
            columns = np.nonzero(edges[1:] > edges[:-1])[0]
            if columns.size:
                starts = edges[columns]
                # Cut at the right edge, or the last column would span the rest of the file.
                visible = edges[-1]
                lows = np.minimum.reduceat(self.peaks_min[:visible], starts)
                highs = np.maximum.reduceat(self.peaks_max[:visible], starts)
                painter.setPen(QPen(QColor(120, 200, 140), 1))
                for x, low, high in zip(columns, lows, highs, strict=True):
                    painter.drawLine(
                        QPointF(x, middle - high * middle), QPointF(x, middle - low * middle)
                    )

        playhead_x = self.x_at(self.playhead)
        painter.setPen(QPen(QColor(255, 80, 80), 2))
        painter.drawLine(QPointF(playhead_x, 0), QPointF(playhead_x, height))
        painter.end()

    def _edge_under(self, x: float):
        for position in range(len(self.cues)):
            cue = self.cues[position]
            if abs(self.x_at(cue.start) - x) <= EDGE_GRAB_PIXELS:
                return position, "start"
            if abs(self.x_at(cue.end) - x) <= EDGE_GRAB_PIXELS:
                return position, "end"
        return None

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        edge = self._edge_under(event.position().x())
        if edge is None:
            self.seek_requested.emit(max(0.0, self.time_at(event.position().x())))
            return
        cue = self.cues[edge[0]]
        self.drag = (edge[0], edge[1], cue.start, cue.end)

    def mouseMoveEvent(self, event) -> None:
        x = event.position().x()
        if self.drag is None:
            if self._edge_under(x) is None:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            else:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            return
        position, edge, start, end = self.drag
        seconds = self.time_at(x)
        # Keep the cue a sensible length and stop it running into its neighbours.
        if edge == "start":
            lowest = 0.0
            if position > 0:
                lowest = self.cues[position - 1].end
            start = min(max(seconds, lowest), end - MIN_CUE_SECONDS)
        else:
            highest = float("inf")
            if position + 1 < len(self.cues):
                highest = self.cues[position + 1].start
            end = max(min(seconds, highest), start + MIN_CUE_SECONDS)
        self.drag = (position, edge, start, end)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self.drag is None:
            return
        position, _edge, start, end = self.drag
        self.drag = None
        cue = self.cues[position]
        if (start, end) != (cue.start, cue.end):
            self.timing_changed.emit(position, start, end)
        self.update()

    def wheelEvent(self, event) -> None:
        anchor = self.time_at(event.position().x())
        factor = 0.8
        if event.angleDelta().y() > 0:
            factor = 1.25
        self.pixels_per_second = min(1000.0, max(5.0, self.pixels_per_second * factor))
        self.view_start = max(0.0, anchor - event.position().x() / self.pixels_per_second)
        self.update()
