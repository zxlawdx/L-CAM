"""Independent recording worker with a bounded queue and monotonic timestamps."""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from time import monotonic

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RecordingFrame:
    captured_at: float
    image: np.ndarray


class RecordingWorker:
    """Recorder runs independently of pywebview preview requests.

    OpenCV VideoWriter produces constant-frame-rate output. Timestamps from
    capture are used to skip excess input frames or duplicate the previous
    frame when processing is slow. This is an *interim video-only* solution;
    audio+video timestamp muxing will require FFmpeg.
    """

    def __init__(self, writer, capacity: int = 90):
        if capacity < 2:
            raise ValueError("Fila de gravação deve comportar pelo menos 2 quadros.")
        self._writer = writer
        self._capacity = capacity
        self._lock = threading.RLock()
        self._queue = None
        self._thread = None
        self._stop = threading.Event()
        self._active = False
        self._fps = 20.0
        self._written = 0
        self._dropped = 0
        self._duplicated = 0
        self._error = None
        self._started_at = None

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._active

    def start(self, path: str, fps: float, width: int, height: int) -> None:
        if type(fps) not in (int, float) or not 1 <= fps <= 60:
            raise ValueError("FPS inválido para gravação.")
        with self._lock:
            if self._active:
                raise RuntimeError("Uma gravação já está em andamento.")
            self._writer.start(path, float(fps), width, height)
            self._queue = queue.Queue(maxsize=self._capacity)
            self._stop.clear()
            self._fps = float(fps)
            self._written = self._dropped = self._duplicated = 0
            self._error = None
            self._started_at = None
            self._active = True
            self._thread = threading.Thread(target=self._run, name="lcam-record", daemon=True)
            self._thread.start()

    def enqueue(self, frame: RecordingFrame) -> None:
        with self._lock:
            if not self._active or self._queue is None:
                return
            target = self._queue
        try:
            target.put_nowait(frame)
        except queue.Full:
            # Never block the video processing worker on disk output.
            try:
                target.get_nowait()
                target.task_done()
            except queue.Empty:
                pass
            with self._lock:
                self._dropped += 1
            try:
                target.put_nowait(frame)
            except queue.Full:
                with self._lock:
                    self._dropped += 1

    def _run(self) -> None:
        base = None
        last_frame = None
        try:
            while True:
                target = self._queue
                if target is None:
                    break
                try:
                    sample = target.get(timeout=0.1)
                except queue.Empty:
                    if self._stop.is_set():
                        break
                    continue
                try:
                    if base is None:
                        base = sample.captured_at
                        with self._lock:
                            self._started_at = base
                    target_index = max(0, round((sample.captured_at - base) * self._fps))
                    # A bounded backlog cannot demand an unbounded catch-up.
                    if last_frame is not None and target_index > self._written:
                        missing = min(target_index - self._written, int(self._fps * 5))
                        for _ in range(missing):
                            self._writer.write(last_frame)
                            self._written += 1
                            self._duplicated += 1
                    if target_index < self._written:
                        # Too many input frames for output FPS.
                        self._dropped += 1
                    else:
                        self._writer.write(sample.image)
                        self._written += 1
                        last_frame = sample.image
                finally:
                    target.task_done()
                if self._stop.is_set() and target.empty():
                    break
        except Exception as exc:
            logger.exception("Falha no worker de gravação.")
            with self._lock:
                self._error = str(exc)
        finally:
            try:
                self._writer.stop()
            except Exception as exc:
                logger.exception("Falha ao finalizar arquivo.")
                with self._lock:
                    self._error = str(exc)
            with self._lock:
                self._active = False

    def stop(self, timeout: float = 8.0) -> dict:
        with self._lock:
            thread = self._thread
            self._stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
            if thread.is_alive():
                raise RuntimeError("Encoder não finalizou dentro do tempo limite.")
        with self._lock:
            return self.stats()

    def stats(self) -> dict:
        with self._lock:
            return {
                "recording": self._active,
                "written_frames": self._written,
                "dropped_record_frames": self._dropped,
                "duplicated_record_frames": self._duplicated,
                "error": self._error,
                "record_fps": self._fps,
                "first_video_frame_at": self._started_at,
                "queued": self._queue.qsize() if self._queue is not None else 0,
            }
