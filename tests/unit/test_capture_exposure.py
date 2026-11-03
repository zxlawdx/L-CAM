"""Camera telemetry reports raw driver controls; it never modifies hardware."""
from __future__ import annotations

import cv2
import numpy as np

from apps.camera.services.capture import CameraCapture
from scripts.diagnose_exposure import diagnose


class FakeVideoCapture:
    def __init__(self):
        self.requests = []
        self.frame = np.zeros((48, 64, 3), dtype=np.uint8)
        self.frame[:, 32:] = 252
        self.read_count = 0
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.requests.append((prop, value))
        return True

    def get(self, prop):
        values = {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 64.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 48.0,
            cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"MJPG")),
            cv2.CAP_PROP_AUTO_EXPOSURE: 0.75,
            cv2.CAP_PROP_EXPOSURE: -6.0,
            cv2.CAP_PROP_GAIN: 10.0,
        }
        return values.get(prop, 0.0)

    def getBackendName(self):
        return "DSHOW"

    def read(self):
        self.read_count += 1
        return True, self.frame.copy()

    def release(self):
        self.released = True


def test_camera_info_is_read_only_and_raw_exposure_is_not_interpreted():
    fake = FakeVideoCapture()
    camera = CameraCapture(index=0, width=64, height=48)
    camera._camera = fake
    recorded = len(fake.requests)
    info = camera.info()
    assert info["active"] is True
    assert info["backend"] == "DSHOW"
    assert info["fourcc"] == "MJPG"
    assert info["driver_reported_fps"] == 30.0
    assert info["auto_exposure_raw"] == 0.75
    assert info["exposure_raw"] == -6.0
    assert len(fake.requests) == recorded


def test_capture_info_when_inactive_is_honest():
    assert CameraCapture().info() == {"active": False}


class FakeSource:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.count = 0

    def start(self):
        self.started = True

    def info(self):
        return {"active": self.started, "backend": "fake",
                "driver_reported_fps": 30.0}

    def read(self):
        self.count += 1
        frame = np.full((24, 32, 3), 55, dtype=np.uint8)
        frame[:, 16:] = 251
        return frame

    def stop(self):
        self.stopped = True


def test_diagnose_exposure_separates_measured_frames_and_camera_settings():
    src = FakeSource()
    result = diagnose(src, frames=12)
    assert src.started and src.stopped
    assert result["frames"] == 12
    assert result["actual_measured_fps"] >= 0
    assert result["near_white_pixels_fraction"] > 0.3
    assert result["dark_pixels_fraction"] > 0.3
    assert "median_read_ms" in result
    assert "notes" in result


def test_capture_diagnostic_releases_camera_after_read_exception():
    class Failing(FakeSource):
        def read(self):
            raise OSError("Simulated driver fault")
    source = Failing()
    try:
        diagnose(source, frames=10)
    except OSError:
        assert source.stopped is True
    else:
        raise AssertionError("Should propagate device error")
