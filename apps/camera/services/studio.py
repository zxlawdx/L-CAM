"""Orquestração do estúdio; independente de Vela e da interface."""
from __future__ import annotations

import base64
import threading
from datetime import datetime
from pathlib import Path

import cv2

from .capture import CameraCapture
from .filters import BasicEnhancer, ImageSettings
from .recorder import VideoRecorder


class CameraStudio:
    """Casos de uso da câmera com acesso serializado ao dispositivo."""

    def __init__(self, camera=None, processor=None, recorder=None):
        self._camera = camera if camera is not None else CameraCapture()
        self._processor = processor if processor is not None else BasicEnhancer()
        self._recorder = recorder if recorder is not None else VideoRecorder()
        self._settings = ImageSettings()
        self._lock = threading.RLock()
        self._active = False
        self._recording_path = None

    def start(self, index: int = 0) -> dict:
        if type(index) is not int or not 0 <= index <= 10:
            raise ValueError("Índice de câmera inválido.")
        with self._lock:
            if self._active:
                self.stop()
            # A implementação de captura deve liberar o dispositivo se falhar.
            self._camera.index = index
            self._camera.start()
            self._active = True
            return {"active": True, "device": index}

    def configure(self, *, brightness=0, contrast=1.0, saturation=1.0, denoise=False):
        if type(denoise) is not bool:
            raise ValueError("denoise deve ser booleano")
        self._settings = ImageSettings(
            brightness=int(brightness),
            contrast=float(contrast),
            saturation=float(saturation),
            denoise=denoise,
        )
        return {"ok": True}

    def frame(self) -> dict:
        with self._lock:
            if not self._active:
                raise RuntimeError("Ative a câmera antes de solicitar quadros.")
            original = self._camera.read()
            processed = self._processor.process(original, self._settings)
            if self._recorder.recording:
                self._recorder.write(processed)
            ok, encoded = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if not ok:
                raise RuntimeError("Não foi possível codificar o quadro.")
            return {
                "image": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"),
                "width": int(processed.shape[1]),
                "height": int(processed.shape[0]),
                "recording": self._recorder.recording,
            }

    def record_start(self, fps: float = 20.0) -> dict:
        if not isinstance(fps, (int, float)) or not 1 <= fps <= 60:
            raise ValueError("FPS fora do intervalo suportado.")
        with self._lock:
            if not self._active:
                raise RuntimeError("Ative a câmera antes de gravar.")
            if self._recorder.recording:
                raise RuntimeError("Uma gravação já está ativa.")
            frame = self._camera.read()
            height, width = frame.shape[:2]
            folder = Path.home() / "Videos" / "L-CAM"
            filename = "lcam-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".mp4"
            path = folder / filename
            self._recorder.start(str(path), float(fps), width, height)
            self._recording_path = str(path)
            return {"recording": True, "path": self._recording_path, "audio": False}

    def record_stop(self) -> dict:
        with self._lock:
            previous = self._recording_path
            self._recorder.stop()
            self._recording_path = None
            return {"recording": False, "path": previous}

    def stop(self) -> dict:
        with self._lock:
            self._recorder.stop()
            self._camera.stop()
            self._active = False
            self._recording_path = None
            return {"active": False}

    def status(self) -> dict:
        with self._lock:
            return {"active": self._active, "recording": self._recorder.recording}

    def close(self) -> None:
        self.stop()
