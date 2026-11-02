"""Orquestração de captura, melhoria e gravação; nenhuma dependência do Vela."""
from __future__ import annotations

import base64
import threading
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2

from .capture import CameraCapture
from .filters import BasicEnhancer, ImageSettings
from .recorder import VideoRecorder


class CameraStudio:
    """Caso de uso principal; hardware, processador e gravação são injetáveis.

    Enquanto a prévia usa chamadas sucessivas da bridge, a gravação continua
    dependente da taxa de quadros solicitados pelo frontend. Um worker separado
    para gravação com sincronização de áudio é uma etapa futura.
    """

    def __init__(self, camera=None, processor=None, recorder=None):
        self._camera = camera if camera is not None else CameraCapture()
        self._processor = processor if processor is not None else BasicEnhancer()
        self._recorder = recorder if recorder is not None else VideoRecorder()
        self._settings = ImageSettings()
        self._lock = threading.RLock()
        self._active = False
        self._recording_path = None
        self._timestamps = deque(maxlen=30)

    def start(self, index: int = 0) -> dict:
        if type(index) is not int or not 0 <= index <= 10:
            raise ValueError("Índice de câmera inválido.")
        with self._lock:
            if self._active:
                self.stop()
            self._camera.index = index
            self._camera.start()
            self._timestamps.clear()
            self._active = True
            return {"active": True, "device": index}

    def configure(
        self,
        *,
        brightness=0,
        contrast=1.0,
        saturation=1.0,
        denoise=False,
        mode="off",
        strength=0.65,
        sharpness=0.0,
    ) -> dict:
        # Não converter strings/bools silenciosamente: bridge recebe dados
        # externos, que devem obedecer aos tipos do contrato.
        settings = ImageSettings(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            denoise=denoise,
            mode=mode,
            strength=strength,
            sharpness=sharpness,
        )
        with self._lock:
            self._settings = settings
            return {"settings": asdict(settings)}

    def frame(self, original_preview: bool = False) -> dict:
        if type(original_preview) is not bool:
            raise ValueError("original_preview deve ser booleano.")
        with self._lock:
            if not self._active:
                raise RuntimeError("Ative a câmera antes de solicitar quadros.")
            started = perf_counter()
            original = self._camera.read()
            processed = self._processor.process(original, self._settings)
            processing_ms = (perf_counter() - started) * 1000.0

            # A comparação é somente visual: gravações continuam aprimoradas.
            if self._recorder.recording:
                self._recorder.write(processed)
            preview = original if original_preview else processed
            ok, encoded = cv2.imencode(
                ".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 86]
            )
            if not ok:
                raise RuntimeError("Não foi possível codificar o quadro.")

            now = perf_counter()
            self._timestamps.append(now)
            if len(self._timestamps) >= 2:
                elapsed = self._timestamps[-1] - self._timestamps[0]
                fps = (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0
            else:
                fps = 0.0
            return {
                "image": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"),
                "width": int(preview.shape[1]),
                "height": int(preview.shape[0]),
                "recording": self._recorder.recording,
                "preview_fps": round(fps, 1),
                "processing_ms": round(processing_ms, 1),
                "mode": self._settings.mode,
            }

    def record_start(self, fps: float = 20.0) -> dict:
        if type(fps) not in (int, float) or not 1 <= fps <= 60:
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
            self._timestamps.clear()
            return {"active": False}

    def status(self) -> dict:
        with self._lock:
            return {
                "active": self._active,
                "recording": self._recorder.recording,
                "settings": asdict(self._settings),
            }

    def close(self) -> None:
        self.stop()
