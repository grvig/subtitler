"""The JSON project file, plus the autosave file kept beside it.

The autosave copy lives next to the project as "<project>.autosave". The
main window writes it every AUTOSAVE_INTERVAL_SECONDS. When a project is
opened, an autosave that is newer than the project means the last session
ended without saving, and the user is offered the autosaved version.
"""

import json
import os
from dataclasses import asdict

import numpy as np

from subtitler.core.models import Cue, Project, Word

FORMAT_VERSION = 1
AUTOSAVE_INTERVAL_SECONDS = 60
AUTOSAVE_SUFFIX = ".autosave"
PEAKS_SUFFIX = ".peaks.npz"


def project_to_dict(project: Project) -> dict:
    return {
        "version": FORMAT_VERSION,
        "media_path": project.media_path,
        "profile": project.profile,
        "shot_boundaries": list(project.shot_boundaries),
        "cues": [asdict(cue) for cue in project.cues],
    }


def project_from_dict(data: dict) -> Project:
    cues = []
    for item in data.get("cues", []):
        words = [Word(**word) for word in item.get("words", [])]
        cues.append(
            Cue(
                index=item["index"],
                start=item["start"],
                end=item["end"],
                lines=list(item["lines"]),
                words=words,
                locked=item.get("locked", False),
            )
        )
    return Project(
        media_path=data.get("media_path"),
        cues=cues,
        shot_boundaries=list(data.get("shot_boundaries", [])),
        profile=data.get("profile", "english"),
    )


def save_project(project: Project, path) -> None:
    # Write to a temporary file first so a crash never leaves half a project.
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(project_to_dict(project), handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def load_project(path) -> Project:
    with open(path, encoding="utf-8") as handle:
        return project_from_dict(json.load(handle))


def autosave_path(path) -> str:
    return f"{path}{AUTOSAVE_SUFFIX}"


def write_autosave(project: Project, path) -> None:
    save_project(project, autosave_path(path))


def find_recovery(path) -> str | None:
    """Return the autosave path if it is newer than the project, else None."""
    candidate = autosave_path(path)
    if not os.path.exists(candidate):
        return None
    if os.path.exists(path) and os.path.getmtime(path) >= os.path.getmtime(candidate):
        return None
    return candidate


def peaks_path(path) -> str:
    return f"{path}{PEAKS_SUFFIX}"


def save_peaks(path, peaks_min: np.ndarray, peaks_max: np.ndarray, buckets_per_second: int) -> None:
    """Waveform peaks go in a compressed numpy file beside the project; an hour
    of audio is far too many numbers for the JSON file."""
    with open(peaks_path(path), "wb") as handle:
        np.savez_compressed(
            handle, peaks_min=peaks_min, peaks_max=peaks_max, rate=buckets_per_second
        )


def load_peaks(path):
    """Return (peaks_min, peaks_max, buckets_per_second), or None if there are none."""
    candidate = peaks_path(path)
    if not os.path.exists(candidate):
        return None
    with np.load(candidate) as data:
        return data["peaks_min"], data["peaks_max"], int(data["rate"])


def discard_autosave(path) -> None:
    candidate = autosave_path(path)
    if os.path.exists(candidate):
        os.remove(candidate)
