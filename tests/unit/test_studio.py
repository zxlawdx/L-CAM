"""Use-case tests with fake camera and writer; no hardware needed."""
from __future__ import annotations

import time
import numpy as np
import pytest

from apps.camera.services.studio import CameraStudio


class FakeCamera:
    index = 0
    started = False

    def __init__(self):
        self.reads = 0

    def start(self):
        self.started = True

    def read(self):
        if not self.started:
            raise RuntimeError("Not started")
        self.reads += 1
        return np.full((24, 32, 3), 25 + (self.reads % 10), dtype=np.uint8)

    def stop(self):
        self.started = False


class FakeRecorder:
    def __init__(self):
        self.recording = False
        self.written = 0
        self.path = None

    def start(self, path, fps, width, height):
        self.recording = True
        self.path = path
        assert (width, height) == (32, 24)

    def write(self, frame):
        self.written += 1

    def stop(self):
        self.recording = False


def wait_until(predicate, timeout=2):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if predicate():
            return True
        time.sleep(0.015)
    return False


def test_start_preview_record_and_stop():
    camera, recorder = FakeCamera(), FakeRecorder()
    studio = CameraStudio(camera=camera, recorder=recorder)
    with pytest.raises(RuntimeError):
        studio.frame()
    assert studio.start()["active"]
    assert wait_until(lambda: studio.status()["pipeline"]["processed_frames"] > 0)
    frame = studio.frame()
    assert frame["image"].startswith("data:image/jpeg;base64,")
    assert frame["sequence"] >= 1
    assert studio.record_start()["recording"]

    # Recording advances with zero additional JS preview requests.
    assert wait_until(lambda: recorder.written >= 2)
    assert camera.reads >= 3
    assert studio.record_stop()["path"] == recorder.path
    assert studio.stop() == {"active": False}
    assert not camera.started


def test_reject_invalid_device_and_inactive_recording():
    studio = CameraStudio(camera=FakeCamera(), recorder=FakeRecorder())
    with pytest.raises(ValueError):
        studio.start(-1)
    with pytest.raises(RuntimeError):
        studio.record_start()


def test_validation_preserves_last_good_settings():
    studio = CameraStudio(camera=FakeCamera(), recorder=FakeRecorder())
    studio.configure(mode="auto", brightness=0)
    with pytest.raises(ValueError):
        studio.configure(brightness=200)
    with pytest.raises(ValueError):
        studio.configure(denoise="true")
    assert studio.status()["settings"]["mode"] == "auto"


def test_preview_requests_do_not_advance_capture():
    camera = FakeCamera()
    studio = CameraStudio(camera=camera, recorder=FakeRecorder())
    studio.start()
    assert wait_until(lambda: studio.status()["pipeline"]["processed_frames"] > 0)
    first = studio.frame()
    studio.frame(last_sequence=first["sequence"])
    # Camera source is driven by its own thread, not by calls to .frame().
    assert studio.status()["pipeline"]["captured_frames"] == camera.reads
    studio.stop()
