import copy

import pytest

from subtitler.core.models import Cue, Project
from subtitler.core.rules.profiles import ENGLISH
from subtitler.core.segmentation import segment
from subtitler.pipeline.asr import sample_words
from subtitler.ui.commands import (
    CommandStack,
    EditTextCommand,
    MergeCuesCommand,
    ResegmentCommand,
    SetTimingCommand,
    SplitCueCommand,
)

PROFILE = ENGLISH


def make_project():
    return Project("clip.mp4", segment(sample_words(), [], PROFILE))


def snapshot(project):
    return copy.deepcopy(project.cues)


def run_and_undo(project, command):
    before = snapshot(project)
    stack = CommandStack()
    stack.execute(command)
    after = snapshot(project)
    assert after != before
    assert project.dirty
    stack.undo()
    assert project.cues == before
    stack.redo()
    assert project.cues == after


def test_edit_text_locks_and_undoes():
    project = make_project()
    run_and_undo(project, EditTextCommand(project, 0, ["Hello there"]))
    assert project.cues[0].lines == ["Hello there"]
    assert project.cues[0].locked


def test_set_timing():
    project = make_project()
    cue = project.cues[1]
    run_and_undo(project, SetTimingCommand(project, 1, cue.start + 0.1, cue.end - 0.1))


def test_split_with_words():
    project = make_project()
    count = len(project.cues)
    cue = project.cues[1]
    middle = cue.words[len(cue.words) // 2].start
    run_and_undo(project, SplitCueCommand(project, 1, middle, PROFILE))
    assert len(project.cues) == count + 1
    first, second = project.cues[1], project.cues[2]
    assert first.end == second.start == middle
    assert f"{first.text()} {second.text()}" == cue.text()
    assert [c.index for c in project.cues] == list(range(1, count + 2))


def test_split_without_words():
    project = Project(cues=[Cue(1, 0.0, 4.0, ["one two three four"])])
    SplitCueCommand(project, 0, 2.0, PROFILE).do()
    assert [c.text() for c in project.cues] == ["one two", "three four"]


def test_split_outside_cue_rejected():
    project = make_project()
    with pytest.raises(ValueError):
        SplitCueCommand(project, 0, project.cues[0].end + 5, PROFILE)


def test_merge():
    project = make_project()
    count = len(project.cues)
    first, second = project.cues[0], project.cues[1]
    run_and_undo(project, MergeCuesCommand(project, 0, PROFILE))
    assert len(project.cues) == count - 1
    assert project.cues[0].start == first.start
    assert project.cues[0].end == second.end
    assert project.cues[0].words == first.words + second.words


def test_merge_last_cue_rejected():
    project = make_project()
    with pytest.raises(ValueError):
        MergeCuesCommand(project, len(project.cues) - 1, PROFILE)


def test_resegment_keeps_edited_cue():
    project = make_project()
    stack = CommandStack()
    stack.execute(EditTextCommand(project, 1, ["An edited", "cue"]))
    edited = copy.deepcopy(project.cues[1])
    stack.execute(ResegmentCommand(project, PROFILE))
    kept = [cue for cue in project.cues if cue.locked]
    assert len(kept) == 1
    assert kept[0].lines == edited.lines
    assert kept[0].start == edited.start
    assert kept[0].end == edited.end
    stack.undo()
    stack.undo()
    assert not any(cue.locked for cue in project.cues)


def test_stack_limit_and_redo_cleared():
    project = make_project()
    stack = CommandStack(limit=3)
    calls = []
    stack.listeners.append(lambda: calls.append(1))
    for number in range(5):
        stack.execute(EditTextCommand(project, 0, [f"text {number}"]))
    assert len(stack.undo_stack) == 3
    stack.undo()
    assert stack.can_redo()
    stack.execute(EditTextCommand(project, 0, ["new"]))
    assert not stack.can_redo()
    assert len(calls) == 7
    assert not CommandStack().undo()
