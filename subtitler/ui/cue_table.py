"""Table of cues. Text edits are reported by signal, not applied here, so the
main window can turn them into undoable EditTextCommands."""

from PySide6.QtCore import QAbstractTableModel, Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableView

from subtitler.core.models import Cue
from subtitler.core.rules.profiles import StyleProfile
from subtitler.io.srt import format_timestamp

COLUMNS = ["#", "Start", "End", "Duration", "CPS", "Speaker", "Text"]
TEXT_COLUMN = 6
CPS_COLUMN = 4
# Lines are shown on one row separated by this, and split on it again after editing.
LINE_SEPARATOR = " | "

CPS_GOOD = QColor(60, 150, 80, 90)
CPS_HIGH = QColor(220, 170, 40, 110)
CPS_TOO_HIGH = QColor(210, 60, 60, 120)


def cue_speakers(cue: Cue) -> str:
    """The speakers of a cue's words, in order, such as "Speaker 1"."""
    names = []
    for word in cue.words:
        if word.speaker and word.speaker not in names:
            names.append(word.speaker)
    return ", ".join(names)


class CueTableModel(QAbstractTableModel):
    text_edited = Signal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cues = []
        self.profile = None

    def set_cues(self, cues: list[Cue], profile: StyleProfile) -> None:
        self.beginResetModel()
        self.cues = cues
        self.profile = profile
        self.endResetModel()

    def rowCount(self, parent=None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return len(self.cues)

    def columnCount(self, parent=None) -> int:
        return len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return COLUMNS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        cue = self.cues[index.row()]
        column = index.column()
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            values = [
                str(cue.index),
                format_timestamp(cue.start, "."),
                format_timestamp(cue.end, "."),
                f"{cue.duration():.2f}",
                f"{cue.chars_per_second():.1f}",
                cue_speakers(cue),
                LINE_SEPARATOR.join(cue.lines),
            ]
            return values[column]
        if role == Qt.ItemDataRole.BackgroundRole and column == CPS_COLUMN:
            cps = cue.chars_per_second()
            if cps > self.profile.max_cps:
                return CPS_TOO_HIGH
            if cps > self.profile.target_cps:
                return CPS_HIGH
            return CPS_GOOD
        if role == Qt.ItemDataRole.FontRole and cue.locked:
            font = QFont()
            font.setItalic(True)
            return font
        if role == Qt.ItemDataRole.ToolTipRole and cue.locked:
            return "Edited by hand: kept as is when re-segmenting"
        return None

    def flags(self, index):
        flags = super().flags(index)
        if index.column() == TEXT_COLUMN:
            flags = flags | Qt.ItemFlag.ItemIsEditable
        return flags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role != Qt.ItemDataRole.EditRole or index.column() != TEXT_COLUMN:
            return False
        if value == self.data(index, role) or not value.strip():
            return False
        lines = [line.strip() for line in value.split(LINE_SEPARATOR.strip())]
        self.text_edited.emit(index.row(), "\n".join(line for line in lines if line))
        return True


class CueTable(QTableView):
    cue_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cue_model = CueTableModel(self)
        self.setModel(self.cue_model)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.verticalHeader().setVisible(False)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(TEXT_COLUMN, QHeaderView.ResizeMode.Stretch)
        self.selectionModel().currentRowChanged.connect(self._on_row)

    def _on_row(self, current, previous) -> None:
        if current.isValid():
            self.cue_selected.emit(current.row())

    def selected_position(self) -> int:
        index = self.currentIndex()
        if index.isValid():
            return index.row()
        return -1

    def select_cue(self, position: int) -> None:
        """Select a row without emitting cue_selected, which is kept for clicks
        by the user, so that selecting from code never seeks the video."""
        if 0 <= position < self.cue_model.rowCount():
            self.blockSignals(True)
            self.selectRow(position)
            self.blockSignals(False)
            self.scrollTo(self.cue_model.index(position, 0))
