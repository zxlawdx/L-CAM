"""Original image comparison never changes the frames sent to recorder."""
from __future__ import annotations

import base64
import time

import cv2
import numpy as np
import pytest

from apps.camera.services.studio import CameraStudio


class FakeCamera:
    index = 0

    def start(self):
        self.running = True

    def read(self):
        return np.full((40, 48, 3), 30, dtype=np.uint8)

    def stop(self):
        self.running = False


class FakeProcessor:
    def process(self, frame, settings):
        return np.full_like(frame, 180)


class FakeRecorder:
    def __init__(self):
        self.recording = False
        self.written = []

    def start(self, path, fps, width, height):
        self.recording = True

    def write(self, frame):
        self.written.append(frame.copy())

    def stop(self):
        self.recording = False


def decode(data_url):
    payload = base64.b64decode(data_url.partition(",")[2])
    return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)


def wait_frame(studio):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        result = studio.frame()
        if "image" in result:
            return result
    raise AssertionError("Preview não foi publicado.")


def test_comparison_uses_raw_preview_and_enhanced_recorder():
    recorder = FakeRecorder()
    studio = CameraStudio(camera=FakeCamera(), processor=FakeProcessor(), recorder=recorder)
    studio.start()
    initial = wait_frame(studio)
    raw = studio.frame(original_preview=True)
    enhanced = studio.frame(original_preview=False)
    assert np.mean(decode(raw["image"])) < 40
    assert np.mean(decode(enhanced["image"])) > 170
    assert raw["width"] == 48 and raw["height"] == 40
    assert "capture_fps" in initial
    assert "processing_ms" in raw

    studio.record_start()
    deadline = time.monotonic() + 2
    while len(recorder.written) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert len(recorder.written) >= 2
    assert all(np.all(frame == 180) for frame in recorder.written)
    studio.stop()


def test_invalid_preview_is_rejected():
    studio = CameraStudio(camera=FakeCamera(), processor=FakeProcessor(), recorder=FakeRecorder())
    studio.start()
    with pytest.raises(ValueError):
        studio.frame(original_preview="yes")
    with pytest.raises(ValueError):
        studio.frame(last_sequence="1")
    studio.stop()


def test_status_reflects_settings():
    studio = CameraStudio(camera=FakeCamera(), processor=FakeProcessor(), recorder=FakeRecorder())
    result = studio.configure(mode="low_light", strength=0.75, sharpness=0.1, denoise=True)
    assert result["settings"]["mode"] == "low_light"
    assert studio.status()["settings"]["strength"] == 0.75
    with pytest.raises(ValueError):
        studio.configure(mode="unsupported")
    assert studio.status()["settings"]["mode"] == "low_light"
