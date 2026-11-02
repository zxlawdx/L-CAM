"""Image comparison must not affect the frames written to the recorder."""
import base64

import cv2
import numpy as np
import pytest

from apps.camera.services.studio import CameraStudio


class FakeCamera:
    index = 0

    def start(self):
        return None

    def read(self):
        return np.full((40, 48, 3), 30, dtype=np.uint8)

    def stop(self):
        return None


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


def _decode(data_url):
    payload = base64.b64decode(data_url.partition(",")[2])
    return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)


def test_original_comparison_is_only_preview_and_never_recording():
    recorder = FakeRecorder()
    studio = CameraStudio(camera=FakeCamera(), processor=FakeProcessor(), recorder=recorder)
    studio.start()
    studio.record_start()
    raw = studio.frame(original_preview=True)
    enhanced = studio.frame(original_preview=False)
    assert np.mean(_decode(raw["image"])) < 40
    assert np.mean(_decode(enhanced["image"])) > 170
    assert recorder.recording
    assert len(recorder.written) == 2
    assert all(np.all(frame == 180) for frame in recorder.written)
    assert raw["width"] == 48 and raw["height"] == 40
    assert "preview_fps" in raw
    assert "processing_ms" in raw
    studio.stop()


def test_invalid_original_preview_is_rejected():
    studio = CameraStudio(camera=FakeCamera(), processor=FakeProcessor(), recorder=FakeRecorder())
    studio.start()
    with pytest.raises(ValueError):
        studio.frame(original_preview="yes")
    studio.stop()


def test_configuration_is_reflected_in_status():
    studio = CameraStudio(camera=FakeCamera(), processor=FakeProcessor(), recorder=FakeRecorder())
    result = studio.configure(mode="low_light", strength=0.75, sharpness=0.1, denoise=True)
    assert result["settings"]["mode"] == "low_light"
    assert studio.status()["settings"]["strength"] == 0.75
    with pytest.raises(ValueError):
        studio.configure(mode="unsupported")
    assert studio.status()["settings"]["mode"] == "low_light"
