"""Safety regression: simulated GPU failure must preserve the camera worker."""
import time

import numpy as np

from apps.camera.services.filters import ImageSettings
from apps.camera.services.realtime import RealtimePipeline


class Camera:
    running = False
    reads = 0

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def read(self):
        self.reads += 1
        return np.full((24, 32, 3), 80, dtype=np.uint8)


class Classic:
    def process(self, image, settings):
        return image.copy()


class CrashingGpu:
    def process(self, image, settings):
        raise RuntimeError("Processo ONNX encerrou (simulated 0xC0000005)")

    def fallback(self):
        return Classic()


class Recorder:
    frames = 0

    def enqueue(self, item):
        self.frames += 1


def test_gpu_failure_falls_back_and_camera_stays_running():
    camera, recorder, errors = Camera(), Recorder(), []
    pipe = RealtimePipeline(
        camera, CrashingGpu(), recorder, ImageSettings(),
        on_processor_error=errors.append,
    )
    pipe.start()
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and pipe.status()["processed_frames"] < 2:
            time.sleep(0.025)
        assert camera.running
        assert recorder.frames >= 2
        assert len(errors) == 1
        assert "ONNX" in errors[0]
        assert pipe.status()["process_fps"] >= 0
        assert "clássico" in pipe.status()["error"]
    finally:
        pipe.stop()
    assert not camera.running
