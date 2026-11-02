"""Capture and processing workers independent of the Vela/WebView event loop."""
from __future__ import annotations

import logging
import threading
from collections import deque
from time import monotonic

from .mailbox import CapturedFrame, LatestFrameMailbox, ProcessedFrame
from .recording_worker import RecordingFrame

logger = logging.getLogger(__name__)


class RealtimePipeline:
    """Single capture producer, single processing consumer, newest-only preview.

    Both stages have bounded storage (one latest frame each). Recording uses a
    separate bounded writer queue and is never triggered by camera_frame().
    """

    def __init__(self, camera, processor, recording_worker, settings, target_fps=30.0, on_processor_error=None):
        if target_fps <= 0:
            raise ValueError("target_fps deve ser positivo.")
        self._camera = camera
        self._processor = processor
        self._recording = recording_worker
        self._settings = settings
        self._settings_lock = threading.Lock()
        self._target_fps = float(target_fps)
        self._on_processor_error = on_processor_error
        self._input = LatestFrameMailbox()
        self._output = LatestFrameMailbox()
        self._stop = threading.Event()
        self._started = False
        self._threads = []
        self._error = None
        self._capture_count = 0
        self._process_count = 0
        self._skipped = 0
        self._reconnects = 0
        self._capture_times = deque(maxlen=60)
        self._process_times = deque(maxlen=60)
        self._stats_lock = threading.Lock()
        self._needs_reset = False

    @staticmethod
    def _fps(timestamps) -> float:
        if len(timestamps) < 2:
            return 0.0
        span = timestamps[-1] - timestamps[0]
        return round((len(timestamps) - 1) / span, 1) if span > 0 else 0.0

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Pipeline já iniciado.")
        self._camera.start()
        self._stop.clear()
        self._started = True
        self._threads = [
            threading.Thread(target=self._capture_loop, name="lcam-capture", daemon=True),
            threading.Thread(target=self._process_loop, name="lcam-process", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def configure(self, settings) -> None:
        with self._settings_lock:
            old = self._settings
            self._settings = settings
            if old.mode != settings.mode or old.denoise != settings.denoise:
                self._needs_reset = True

    def set_processor(self, processor) -> None:
        """Swap frame processing strategy without touching capture or Vela."""
        with self._settings_lock:
            self._processor = processor
            self._needs_reset = True

    def _capture_loop(self) -> None:
        failures = 0
        sequence = 0
        interval = 1.0 / self._target_fps
        try:
            while not self._stop.is_set():
                started = monotonic()
                try:
                    frame = self._camera.read()
                    if frame is None:
                        raise RuntimeError("Driver retornou um quadro vazio.")
                    failures = 0
                except Exception as exc:
                    failures += 1
                    logger.warning("Falha de captura (%s/3): %s", failures, exc)
                    if failures >= 3:
                        with self._stats_lock:
                            self._error = "Webcam desconectada ou indisponível: " + str(exc)
                        break
                    try:
                        self._camera.stop()
                        if self._stop.wait(0.3):
                            break
                        self._camera.start()
                        with self._stats_lock:
                            self._reconnects += 1
                    except Exception as restart_error:
                        logger.warning("Reconexão falhou: %s", restart_error)
                    continue
                sequence += 1
                self._input.publish(CapturedFrame(sequence, started, frame))
                with self._stats_lock:
                    self._capture_count += 1
                    self._capture_times.append(started)
                # Never let an instant fake frame source busy-loop at 100% CPU.
                self._stop.wait(max(0.0, interval - (monotonic() - started)))
        finally:
            self._input.close()

    def _process_loop(self) -> None:
        last_sequence = -1
        failures = 0
        try:
            while not self._stop.is_set():
                sample = self._input.wait_newer_than(last_sequence, timeout=0.2)
                if sample is None:
                    if self._input.closed:
                        break
                    continue
                if last_sequence >= 0 and sample.sequence > last_sequence + 1:
                    with self._stats_lock:
                        self._skipped += sample.sequence - last_sequence - 1
                last_sequence = sample.sequence
                with self._settings_lock:
                    settings = self._settings
                    reset = self._needs_reset
                    processor = self._processor
                    self._needs_reset = False
                if reset and callable(getattr(processor, "reset", None)):
                    processor.reset()

                started = monotonic()
                try:
                    enhanced = processor.process(sample.image, settings)
                    failures = 0
                except Exception as exc:
                    failures += 1
                    logger.exception("Falha no processamento do quadro.")
                    fallback = getattr(processor, "fallback", None)
                    if callable(fallback):
                        # Safe CPU/classic fallback on any failed ONNX inference.
                        self.set_processor(fallback())
                        if callable(self._on_processor_error):
                            self._on_processor_error(str(exc))
                        with self._stats_lock:
                            self._error = "IA desativada após erro; processamento clássico ativo: " + str(exc)
                        failures = 0
                        continue
                    if failures >= 3:
                        with self._stats_lock:
                            self._error = "Falha contínua de processamento: " + str(exc)
                        self._stop.set()
                        break
                    continue

                completed = monotonic()
                packet = ProcessedFrame(
                    sequence=sample.sequence,
                    captured_at=sample.captured_at,
                    completed_at=completed,
                    original=sample.image,
                    image=enhanced,
                    processing_ms=(completed - started) * 1000.0,
                )
                self._output.publish(packet)
                self._recording.enqueue(RecordingFrame(sample.captured_at, enhanced))
                with self._stats_lock:
                    self._process_count += 1
                    self._process_times.append(completed)
        finally:
            self._output.close()

    def next_preview(self, sequence=-1, timeout=0.35):
        return self._output.wait_newer_than(sequence, timeout=timeout)

    def latest(self):
        return self._output.latest()

    def status(self) -> dict:
        with self._stats_lock:
            result = {
                "captured_frames": self._capture_count,
                "processed_frames": self._process_count,
                "dropped_preprocessing": self._skipped,
                "capture_fps": self._fps(self._capture_times),
                "process_fps": self._fps(self._process_times),
                "reconnects": self._reconnects,
                "error": self._error,
                "worker_alive": {thread.name: thread.is_alive() for thread in self._threads},
            }
        return result

    def stop(self) -> None:
        self._stop.set()
        self._input.close()
        # Capture worker may be inside blocking driver.read; do not hold any
        # main/UI lock while waiting on worker shutdown.
        for thread in self._threads:
            if thread is threading.current_thread():
                continue
            thread.join(timeout=2.0)
        try:
            self._camera.stop()
        finally:
            for thread in self._threads:
                if thread is not threading.current_thread() and thread.is_alive():
                    thread.join(timeout=1.5)
            self._output.close()
            self._started = False
