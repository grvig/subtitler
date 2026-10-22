"""Shot change detection with PySceneDetect."""

import threading
from collections.abc import Callable

# Seconds of video handed to the detector at a time, so progress can be
# reported and a cancel request noticed between chunks.
CHUNK_SECONDS = 10.0


def detect_shots(
    media_path,
    progress_callback: Callable[[float], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> list[float]:
    """Times in seconds where the picture cuts to a new shot. Audio-only files
    and files that cannot be decoded give an empty list."""
    # Imported here so that OpenCV is only loaded when detection is used.
    from scenedetect import ContentDetector, SceneManager, open_video
    from scenedetect.video_stream import VideoOpenFailure

    try:
        video = open_video(str(media_path))
    except (VideoOpenFailure, OSError):
        return []
    total = video.duration.seconds
    if total <= 0:
        return []
    manager = SceneManager()
    manager.auto_downscale = True
    manager.add_detector(ContentDetector())
    position = 0.0
    while position < total:
        if cancel_event is not None and cancel_event.is_set():
            break
        position = min(total, position + CHUNK_SECONDS)
        processed = manager.detect_scenes(video, end_time=position)
        if processed == 0:
            break
        if progress_callback is not None:
            progress_callback(100.0 * position / total)
    # Every scene after the first one begins on a cut.
    scenes = manager.get_scene_list()
    return [start.seconds for start, _end in scenes[1:]]
