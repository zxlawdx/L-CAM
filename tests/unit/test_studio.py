"""Testes de casos de uso sem webcam nem arquivo físico."""
import numpy as np
import pytest

from apps.camera.services.studio import CameraStudio


class FakeCamera:
    index = 0
    started = False

    def start(self):
        self.started = True

    def read(self):
        if not self.started:
            raise RuntimeError("Not started")
        return np.zeros((24, 32, 3), dtype=np.uint8)

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


def test_start_frame_record_and_stop():
    camera, recorder = FakeCamera(), FakeRecorder()
    studio = CameraStudio(camera=camera, recorder=recorder)
    with pytest.raises(RuntimeError):
        studio.frame()
    assert studio.start()["active"]
    assert studio.frame()["image"].startswith("data:image/jpeg;base64,")
    assert studio.record_start()["recording"]
    studio.frame()
    assert recorder.written == 1
    assert studio.record_stop()["path"] == recorder.path
    assert studio.stop() == {"active": False}
    assert not camera.started


def test_reject_invalid_camera_and_recording_without_preview():
    studio = CameraStudio(camera=FakeCamera(), recorder=FakeRecorder())
    with pytest.raises(ValueError):
        studio.start(-1)
    with pytest.raises(RuntimeError):
        studio.record_start()


def test_settings_validate():
    studio = CameraStudio(camera=FakeCamera(), recorder=FakeRecorder())
    with pytest.raises(ValueError):
        studio.configure(brightness=200)
    with pytest.raises(ValueError):
        studio.configure(denoise="true")
