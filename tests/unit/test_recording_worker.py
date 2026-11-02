"""The recording queue is independent of the preview loop."""
import time
import numpy as np

from apps.camera.services.recording_worker import RecordingFrame, RecordingWorker


class MemoryWriter:
    def __init__(self):
        self.recording = False
        self.frames = []
        self.path = ""
    def start(self, path, fps, width, height):
        self.path = path
        self.recording = True
    def write(self, frame):
        self.frames.append(frame.copy())
    def stop(self):
        self.recording = False


def test_recording_independent_of_preview_and_timestamp_pacing():
    writer = MemoryWriter()
    worker = RecordingWorker(writer, capacity=3)
    worker.start("test.mp4", 20, 16, 8)
    start = time.monotonic()
    image = np.ones((8, 16, 3), dtype=np.uint8)
    worker.enqueue(RecordingFrame(start, image))
    time.sleep(0.05)
    worker.enqueue(RecordingFrame(start + 0.15, image * 2))
    stats = worker.stop()
    assert not writer.recording
    assert len(writer.frames) >= 2
    assert stats["written_frames"] == len(writer.frames)
    assert stats["duplicated_record_frames"] >= 1


def test_recording_has_bounded_queue():
    writer = MemoryWriter()
    worker = RecordingWorker(writer, capacity=3)
    worker.start("test.mp4", 20, 16, 8)
    frame = np.ones((8, 16, 3), dtype=np.uint8)
    now = time.monotonic()
    for n in range(500):
        worker.enqueue(RecordingFrame(now + n / 20, frame))
    worker.stop(timeout=3)
    assert worker.stats()["queued"] <= 3
