"""List of rule violations grouped by severity."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

from subtitler.core.models import Violation
from subtitler.io.srt import format_timestamp

SEVERITY_ORDER = ["error", "warning", "info"]


class LintPanel(QTreeWidget):
    violation_activated = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Rule", "Cue", "Time", "Message"])
        self.setRootIsDecorated(True)
        self.setAlternatingRowColors(True)
        self.itemDoubleClicked.connect(self._on_double_click)

    def set_violations(self, violations: list[Violation]) -> None:
        # Remember which groups the user had collapsed so a refresh keeps them.
        collapsed = set()
        for number in range(self.topLevelItemCount()):
            item = self.topLevelItem(number)
            if not item.isExpanded():
                collapsed.add(item.data(0, Qt.ItemDataRole.UserRole))
        self.clear()
        for severity in SEVERITY_ORDER:
            matching = [v for v in violations if v.severity == severity]
            if not matching:
                continue
            group = QTreeWidgetItem([f"{severity.title()}s ({len(matching)})"])
            group.setData(0, Qt.ItemDataRole.UserRole, severity)
            self.addTopLevelItem(group)
            for violation in matching:
                row = QTreeWidgetItem(
                    [
                        violation.rule_id,
                        str(violation.cue_index),
                        format_timestamp(violation.timecode, "."),
                        violation.message,
                    ]
                )
                row.setData(0, Qt.ItemDataRole.UserRole, violation)
                group.addChild(row)
            group.setExpanded(severity not in collapsed)
        for column in range(3):
            self.resizeColumnToContents(column)

    def _on_double_click(self, item, column) -> None:
        violation = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(violation, Violation):
            self.violation_activated.emit(violation)
