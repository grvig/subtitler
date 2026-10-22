"""Undoable edits to a project's cue list.

Every change the user makes to cues goes through a Command so it can be
undone. Commands only touch the Project, never widgets; the main window
listens to the CommandStack and refreshes the views after each change.
"""

from dataclasses import replace

from subtitler.core.linebreak import best_split
from subtitler.core.models import Cue, Project
from subtitler.core.rules.profiles import StyleProfile
from subtitler.core.segmentation import segment

DEFAULT_UNDO_LIMIT = 200


def layout_lines(text: str, profile: StyleProfile) -> list[str]:
    """Lines for text: the user's own line breaks if there are any,
    otherwise the best automatic split, otherwise a single line."""
    if "\n" in text:
        return [line.strip() for line in text.split("\n") if line.strip()]
    split = best_split(text, profile)
    if split is None:
        return [" ".join(text.split())]
    return split.lines


def renumber(cues: list[Cue]) -> None:
    for number, cue in enumerate(cues, start=1):
        cue.index = number


class Command:
    description = "Edit"

    def do(self) -> None:
        raise NotImplementedError

    def undo(self) -> None:
        raise NotImplementedError


class EditTextCommand(Command):
    """Replace a cue's text. The cue is locked so re-segmenting keeps it."""

    description = "Edit text"

    def __init__(self, project: Project, position: int, lines: list[str]):
        self.project = project
        self.position = position
        self.new_lines = list(lines)
        self.old_lines = list(project.cues[position].lines)
        self.old_locked = project.cues[position].locked

    def do(self) -> None:
        cue = self.project.cues[self.position]
        cue.lines = list(self.new_lines)
        cue.locked = True

    def undo(self) -> None:
        cue = self.project.cues[self.position]
        cue.lines = list(self.old_lines)
        cue.locked = self.old_locked


class SetTimingCommand(Command):
    description = "Change timing"

    def __init__(self, project: Project, position: int, start: float, end: float):
        self.project = project
        self.position = position
        self.new_times = (start, end)
        cue = project.cues[position]
        self.old_times = (cue.start, cue.end)

    def do(self) -> None:
        cue = self.project.cues[self.position]
        cue.start, cue.end = self.new_times

    def undo(self) -> None:
        cue = self.project.cues[self.position]
        cue.start, cue.end = self.old_times


class SplitCueCommand(Command):
    """Split one cue into two at a time inside it."""

    description = "Split cue"

    def __init__(self, project: Project, position: int, at_time: float, profile: StyleProfile):
        self.project = project
        self.position = position
        self.original = project.cues[position]
        cue = self.original
        if not cue.start < at_time < cue.end:
            raise ValueError("The split point must fall inside the cue")
        if cue.words:
            first_words = [w for w in cue.words if (w.start + w.end) / 2 < at_time]
            second_words = cue.words[len(first_words) :]
            first_text = " ".join(w.text for w in first_words)
            second_text = " ".join(w.text for w in second_words)
        else:
            # No word timings (an imported file): split the words in proportion to time.
            tokens = cue.text().split()
            count = round(len(tokens) * (at_time - cue.start) / cue.duration())
            first_words = []
            second_words = []
            first_text = " ".join(tokens[:count])
            second_text = " ".join(tokens[count:])
        if not first_text or not second_text:
            raise ValueError("Both halves of a split need some text")
        self.parts = [
            Cue(0, cue.start, at_time, layout_lines(first_text, profile), first_words),
            Cue(0, at_time, cue.end, layout_lines(second_text, profile), second_words),
        ]

    def do(self) -> None:
        self.project.cues[self.position : self.position + 1] = self.parts
        renumber(self.project.cues)

    def undo(self) -> None:
        self.project.cues[self.position : self.position + 2] = [self.original]
        renumber(self.project.cues)


class MergeCuesCommand(Command):
    """Merge a cue with the one after it."""

    description = "Merge cues"

    def __init__(self, project: Project, position: int, profile: StyleProfile):
        if position + 1 >= len(project.cues):
            raise ValueError("There is no next cue to merge with")
        self.project = project
        self.position = position
        first = project.cues[position]
        second = project.cues[position + 1]
        self.originals = [first, second]
        text = f"{first.text()} {second.text()}"
        self.merged = Cue(
            0,
            first.start,
            second.end,
            layout_lines(text, profile),
            first.words + second.words,
            first.locked or second.locked,
        )

    def do(self) -> None:
        self.project.cues[self.position : self.position + 2] = [self.merged]
        renumber(self.project.cues)

    def undo(self) -> None:
        self.project.cues[self.position : self.position + 1] = self.originals
        renumber(self.project.cues)


class RealignCommand(Command):
    """Give cues new word timings for their current text. Once the words match
    the text the cue no longer needs its lock: re-segmenting will rebuild it
    from the corrected words."""

    description = "Realign cues"

    def __init__(self, project: Project, results: list[tuple[int, float, list]]):
        self.project = project
        self.changes = []
        for position, start, words in results:
            # Skip a cue that was moved or removed while it was being realigned.
            if position < len(project.cues) and project.cues[position].start == start:
                cue = project.cues[position]
                self.changes.append((position, cue.words, cue.locked, list(words)))

    def do(self) -> None:
        for position, _old_words, _old_locked, new_words in self.changes:
            cue = self.project.cues[position]
            cue.words = list(new_words)
            cue.locked = False

    def undo(self) -> None:
        for position, old_words, old_locked, _new_words in self.changes:
            cue = self.project.cues[position]
            cue.words = old_words
            cue.locked = old_locked


class ResegmentCommand(Command):
    """Run the segmenter again over all words, keeping locked cues as they are."""

    description = "Re-segment"

    def __init__(self, project: Project, profile: StyleProfile):
        self.project = project
        self.profile = profile
        self.old_cues = list(project.cues)
        self.new_cues = None

    def do(self) -> None:
        if self.new_cues is None:
            words = [word for cue in self.old_cues for word in cue.words]
            # Locked cues are copied so later edits cannot reach into the undo history.
            existing = [replace(cue) for cue in self.old_cues]
            self.new_cues = segment(words, self.project.shot_boundaries, self.profile, existing)
        self.project.cues = list(self.new_cues)

    def undo(self) -> None:
        self.project.cues = list(self.old_cues)


class CommandStack:
    def __init__(self, limit: int = DEFAULT_UNDO_LIMIT):
        self.limit = limit
        self.undo_stack = []
        self.redo_stack = []
        self.listeners = []

    def _changed(self) -> None:
        for listener in self.listeners:
            listener()

    def execute(self, command: Command) -> None:
        command.do()
        self.undo_stack.append(command)
        if len(self.undo_stack) > self.limit:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self._mark_dirty(command)
        self._changed()

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        command = self.undo_stack.pop()
        command.undo()
        self.redo_stack.append(command)
        self._mark_dirty(command)
        self._changed()
        return True

    def redo(self) -> bool:
        if not self.redo_stack:
            return False
        command = self.redo_stack.pop()
        command.do()
        self.undo_stack.append(command)
        self._mark_dirty(command)
        self._changed()
        return True

    def can_undo(self) -> bool:
        return bool(self.undo_stack)

    def can_redo(self) -> bool:
        return bool(self.redo_stack)

    def clear(self) -> None:
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._changed()

    def _mark_dirty(self, command: Command) -> None:
        command.project.dirty = True
