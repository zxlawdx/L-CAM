"""Device tests use fake OpenCV handles, no real webcam."""
import cv2

from apps.camera.services.capture import CameraCapture


class FakeVideoCapture:
    def __init__(self, opened=True, readable=True):
        self.opened = opened
        self.readable = readable
        self.set_properties = []
        self.released = False

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.set_properties.append((prop, value))
        return True

    def read(self):
        if not self.readable:
            return False, None
        import numpy as np
        return True, np.zeros((720, 1280, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def test_capture_requests_preferred_resolution(monkeypatch):
    fake = FakeVideoCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: fake)
    camera = CameraCapture(width=1280, height=720, fps=30)
    camera.start()
    assert (cv2.CAP_PROP_FRAME_WIDTH, 1280) in fake.set_properties
    assert (cv2.CAP_PROP_FRAME_HEIGHT, 720) in fake.set_properties
    assert (cv2.CAP_PROP_FPS, 30) in fake.set_properties
    assert camera.read().shape == (720, 1280, 3)
    camera.stop()
    assert fake.released


def test_capture_releases_failed_devices(monkeypatch):
    handles = []
    def fake_constructor(*args):
        handle = FakeVideoCapture(readable=False)
        handles.append(handle)
        return handle
    monkeypatch.setattr(cv2, "VideoCapture", fake_constructor)
    import pytest
    with pytest.raises(RuntimeError):
        CameraCapture().start()
    assert all(handle.released for handle in handles)
