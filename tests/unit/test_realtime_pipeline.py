"""Concurrency and bounded mailbox behavior without physical hardware."""
from __future__ import annotations

import time
import threading
import numpy as np

from apps.camera.services.mailbox import CapturedFrame, LatestFrameMailbox
from apps.camera.services.realtime import RealtimePipeline
from apps.camera.services.filters import ImageSettings


def wait_until(predicate, timeout=1.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.012)
    return False


def test_mailbox_keeps_only_latest_frame():
    box = LatestFrameMailbox()
    for n in range(100):
        box.publish(CapturedFrame(n, time.monotonic(), np.zeros((4, 4, 3), dtype=np.uint8)))
    result = box.wait_newer_than(-1)
    assert result.sequence == 99
    assert box.published == 100
    assert box.latest().sequence == 99
    box.close()
    assert box.closed


def test_mailbox_waiter_wakes_on_publication():
    box = LatestFrameMailbox()
    answer = []
    thread = threading.Thread(target=lambda: answer.append(box.wait_newer_than(4, 0.8)))
    thread.start()
    box.publish(CapturedFrame(5, time.monotonic(), np.zeros((4, 4, 3), dtype=np.uint8)))
    thread.join(timeout=1)
    assert answer[0].sequence == 5


class FakeCapture:
    def __init__(self):
        self.index = 0
        self.running = False
        self.reads = 0
    def start(self):
        self.running = True
    def read(self):
        if not self.running:
            raise RuntimeError("closed")
        self.reads += 1
        return np.full((16, 16, 3), self.reads % 255, dtype=np.uint8)
    def stop(self):
        self.running = False


class SlowProcessor:
    def __init__(self):
        self.calls = 0
    def process(self, frame, settings):
        self.calls += 1
        time.sleep(0.11)
        return frame.copy()


class FakeSink:
    def __init__(self):
        self.frames = 0
    def enqueue(self, packet):
        self.frames += 1


def test_pipeline_drops_stale_preview_frames_and_stops_all_workers():
    camera, processor, sink = FakeCapture(), SlowProcessor(), FakeSink()
    pipeline = RealtimePipeline(camera, processor, sink, ImageSettings(), target_fps=30)
    pipeline.start()
    assert wait_until(lambda: pipeline.status()["processed_frames"] >= 3)
    stats = pipeline.status()
    assert stats["captured_frames"] > stats["processed_frames"]
    assert stats["dropped_preprocessing"] >= 1
    assert sink.frames >= 3
    pipeline.stop()
    assert not camera.running
    assert all(not value for value in pipeline.status()["worker_alive"].values())


def test_no_unbounded_work_when_frame_source_is_fast():
    capture, processor, sink = FakeCapture(), SlowProcessor(), FakeSink()
    pipeline = RealtimePipeline(capture, processor, sink, ImageSettings(), target_fps=120)
    pipeline.start()
    assert wait_until(lambda: pipeline.status()["captured_frames"] >= 15)
    assert pipeline._input.published <= pipeline.status()["captured_frames"]
    assert pipeline._input.latest() is not None
    pipeline.stop()
