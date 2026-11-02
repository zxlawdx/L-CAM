"""Hardware capture adapter; no Vela dependency."""
from __future__ import annotations
import threading
import cv2

class CameraCapture:
    def __init__(self, index: int = 0):
        self.index = index
        self._camera = None
        self._lock = threading.RLock()

    def start(self) -> None:
        with self._lock:
            if self._camera is not None:
                return
            camera = cv2.VideoCapture(self.index, cv2.CAP_DSHOW if __import__("sys").platform == "win32" else cv2.CAP_ANY)
            if not camera.isOpened():
                camera.release()
                raise RuntimeError(f"Não foi possível abrir a câmera {self.index}")
            self._camera = camera

    def read(self):
        with self._lock:
            if self._camera is None:
                raise RuntimeError("Câmera não iniciada")
            ok, frame = self._camera.read()
            if not ok:
                raise RuntimeError("Falha ao capturar quadro")
            return frame

    def stop(self) -> None:
        with self._lock:
            if self._camera is not None:
                self._camera.release()
                self._camera = None
