"""OpenCV camera adapter: device lifecycle and resource ownership."""
from __future__ import annotations

import sys
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
            # CAP_DSHOW é geralmente mais estável no Windows, mas nem todo
            # driver oferece suporte: tentamos o backend padrão como fallback.
            backends = (cv2.CAP_DSHOW, cv2.CAP_ANY) if sys.platform == "win32" else (cv2.CAP_ANY,)
            for backend in backends:
                camera = cv2.VideoCapture(self.index, backend)
                if camera.isOpened():
                    self._camera = camera
                    return
                camera.release()
            raise RuntimeError(f"Não foi possível abrir a câmera {self.index}.")

    def read(self):
        with self._lock:
            if self._camera is None:
                raise RuntimeError("Câmera não iniciada.")
            ok, frame = self._camera.read()
            if not ok or frame is None:
                raise RuntimeError("Falha ao capturar um quadro da câmera.")
            return frame

    def stop(self) -> None:
        with self._lock:
            if self._camera is not None:
                self._camera.release()
                self._camera = None
