"""CameraStudio use cases: Vela-independent camera, preview and recording."""
from __future__ import annotations

import base64
import threading
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import monotonic

from .capture import CameraCapture
from .contracts import CameraSource, FrameProcessor, PreviewEncoder, RecordingSink
from .encoding import JpegPreviewEncoder
from .filters import BasicEnhancer, ImageSettings
from .realtime import RealtimePipeline
from .model_manager import AsyncModelManager
from .recorder import VideoRecorder
from .recording_worker import RecordingWorker


class CameraStudio:
    """Coordinates separate capture, processing and recording workers.

    Capture and processing never depend on how frequently JavaScript renders.
    Instantiating a different FrameProcessor or PreviewEncoder requires no
    changes in the Vela bridge or the application UI.
    """

    def __init__(
        self,
        camera: CameraSource | None = None,
        processor: FrameProcessor | None = None,
        recorder: RecordingSink | None = None,
        encoder: PreviewEncoder | None = None,
    ):
        self._camera = camera if camera is not None else CameraCapture()
        self._processor = processor if processor is not None else BasicEnhancer()
        self._recorder = recorder if recorder is not None else VideoRecorder()
        self._encoder = encoder if encoder is not None else JpegPreviewEncoder(86)
        self._recording = RecordingWorker(self._recorder)
        self._pipeline = None
        self._settings = ImageSettings()
        self._lock = threading.RLock()
        self._recording_path = None
        self._preview_times = deque(maxlen=45)
        self._models = AsyncModelManager(on_ready=self._install_processor)

    def _install_processor(self, processor) -> None:
        with self._lock:
            self._processor = processor
            if self._pipeline is not None:
                self._pipeline.set_processor(processor)

    def model_load(self, manifest_path: str, provider: str = "auto") -> dict:
        return self._models.load(manifest_path, provider)

    def model_unload(self) -> dict:
        return self._models.unload()

    def model_status(self) -> dict:
        return self._models.status()

    @staticmethod
    def _fps(times) -> float:
        if len(times) < 2:
            return 0.0
        duration = times[-1] - times[0]
        return round((len(times) - 1) / duration, 1) if duration > 0 else 0.0

    def start(self, index: int = 0) -> dict:
        if type(index) is not int or not 0 <= index <= 10:
            raise ValueError("Índice de câmera inválido.")
        with self._lock:
            if self._pipeline is not None:
                self.stop()
            self._camera.index = index
            pipeline = RealtimePipeline(
                camera=self._camera,
                processor=self._processor,
                recording_worker=self._recording,
                settings=self._settings,
                on_processor_error=self._models.report_runtime_error,
            )
            pipeline.start()
            self._pipeline = pipeline
            self._preview_times.clear()
            return {"active": True, "device": index, "workers": ["capture", "process"]}

    def configure(
        self, *, brightness=0, contrast=1.0, saturation=1.0,
        denoise=False, mode="off", strength=0.65, sharpness=0.0
    ) -> dict:
        # Explicit immutable validation before updating the active pipeline.
        settings = ImageSettings(
            brightness=brightness, contrast=contrast,
            saturation=saturation, denoise=denoise,
            mode=mode, strength=strength, sharpness=sharpness,
        )
        with self._lock:
            self._settings = settings
            if self._pipeline is not None:
                self._pipeline.configure(settings)
            return {"settings": asdict(settings)}

    def frame(self, original_preview: bool = False, last_sequence: int = -1) -> dict:
        if type(original_preview) is not bool:
            raise ValueError("original_preview deve ser booleano.")
        if type(last_sequence) is not int or last_sequence < -1:
            raise ValueError("last_sequence deve ser inteiro positivo ou -1.")
        with self._lock:
            pipeline = self._pipeline
        if pipeline is None:
            raise RuntimeError("Ative a câmera antes de solicitar quadros.")

        # Wait for new processed pixels without driving capture/processing.
        packet = pipeline.next_preview(last_sequence, timeout=0.25)
        if packet is None:
            state = pipeline.status()
            if state["error"] and not state["worker_alive"].get("lcam-process", False):
                raise RuntimeError(state["error"])
            latest = pipeline.latest()
            if latest is None:
                state = pipeline.status()
                if state["error"]:
                    raise RuntimeError(state["error"])
                return {"waiting": True, "sequence": last_sequence}
            if latest.sequence <= last_sequence:
                return {"unchanged": True, "sequence": last_sequence}
            packet = latest

        selected = packet.original if original_preview else packet.image
        jpeg = self._encoder.encode(selected)
        now = monotonic()
        with self._lock:
            self._preview_times.append(now)
            preview_fps = self._fps(self._preview_times)
        telemetry = pipeline.status()
        return {
            "image": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"),
            "sequence": packet.sequence,
            "width": int(selected.shape[1]),
            "height": int(selected.shape[0]),
            "recording": self._recording.recording,
            "preview_fps": preview_fps,
            "capture_fps": telemetry["capture_fps"],
            "process_fps": telemetry["process_fps"],
            "processing_ms": round(packet.processing_ms, 1),
            "latency_ms": round((now - packet.captured_at) * 1000, 1),
            "dropped_frames": telemetry["dropped_preprocessing"],
            "mode": self._settings.mode,
        }

    def record_start(self, fps: float = 30.0) -> dict:
        with self._lock:
            pipeline = self._pipeline
            if pipeline is None:
                raise RuntimeError("Ative a câmera antes de gravar.")
            if self._recording.recording:
                raise RuntimeError("Uma gravação já está ativa.")
        packet = pipeline.latest() or pipeline.next_preview(-1, timeout=1.0)
        if packet is None:
            raise RuntimeError("Aguarde o primeiro quadro antes de gravar.")
        height, width = packet.image.shape[:2]
        folder = Path.home() / "Videos" / "L-CAM"
        filename = "lcam-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".mp4"
        path = folder / filename
        self._recording.start(str(path), float(fps), width, height)
        with self._lock:
            self._recording_path = str(path)
        return {"recording": True, "path": str(path), "audio": False, "writer": "async"}

    def record_stop(self) -> dict:
        result = self._recording.stop()
        with self._lock:
            path = self._recording_path
            self._recording_path = None
        return {"recording": False, "path": path, **result}

    def stop(self) -> dict:
        with self._lock:
            pipeline = self._pipeline
            self._pipeline = None
        if pipeline is not None:
            pipeline.stop()
        if self._recording.recording:
            self._recording.stop()
        with self._lock:
            self._recording_path = None
            self._preview_times.clear()
        return {"active": False}

    def status(self) -> dict:
        with self._lock:
            pipeline = self._pipeline
            settings = asdict(self._settings)
            path = self._recording_path
        return {
            "active": pipeline is not None,
            "recording": self._recording.recording,
            "settings": settings,
            "recording_path": path,
            "pipeline": pipeline.status() if pipeline is not None else None,
            "writer": self._recording.stats(),
            "model": self._models.status(),
        }

    def close(self) -> None:
        self._models.close()
        self.stop()
