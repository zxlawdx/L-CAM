"""Thread-safe newest-frame mailboxes and timestamped camera samples.

Never keep an unbounded frame history: capture wins over outdated preview.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from time import monotonic

import numpy as np


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    sequence: int
    captured_at: float
    image: np.ndarray


@dataclass(frozen=True, slots=True)
class ProcessedFrame:
    sequence: int
    captured_at: float
    completed_at: float
    original: np.ndarray
    image: np.ndarray
    processing_ms: float


class LatestFrameMailbox:
    """Single-slot mailbox with a condition variable.

    A slow consumer waits for the newest frame and skips older ones, rather
    than buffering increasingly stale camera images.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._frame = None
        self._closed = False
        self.published = 0

    def publish(self, frame) -> None:
        with self._condition:
            if self._closed:
                return
            self._frame = frame
            self.published += 1
            self._condition.notify_all()

    def wait_newer_than(self, sequence: int, timeout: float = 0.25):
        if type(sequence) is not int:
            raise ValueError("sequence deve ser um inteiro.")
        deadline = monotonic() + max(0.0, float(timeout))
        with self._condition:
            while not self._closed:
                frame = self._frame
                if frame is not None and frame.sequence > sequence:
                    return frame
                left = deadline - monotonic()
                if left <= 0:
                    break
                self._condition.wait(left)
            frame = self._frame
            if frame is not None and frame.sequence > sequence:
                return frame
            return None

    def latest(self):
        with self._condition:
            return self._frame

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
