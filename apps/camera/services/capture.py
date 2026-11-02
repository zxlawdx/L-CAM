"""Adaptador OpenCV de captura nativa, com negociação de resolução."""
from __future__ import annotations

import sys
import threading
import cv2


class CameraCapture:
    """Gerencia exclusivamente o dispositivo, sem conhecer Vela ou a interface."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.index = index
        self.preferred_size = (width, height)
        self.preferred_fps = fps
        self._camera = None
        self._lock = threading.RLock()

    @staticmethod
    def _probe(camera) -> bool:
        # Alguns drivers precisam de mais de um read para inicializar.
        for _ in range(3):
            ok, frame = camera.read()
            if ok and frame is not None:
                return True
        return False

    def start(self) -> None:
        with self._lock:
            if self._camera is not None:
                return
            backends = (
                (cv2.CAP_DSHOW, cv2.CAP_ANY)
                if sys.platform == "win32" else (cv2.CAP_ANY,)
            )
            for backend in backends:
                camera = cv2.VideoCapture(self.index, backend)
                try:
                    if not camera.isOpened():
                        continue
                    # Solicita resolução nativa superior. O driver decide se
                    # suporta; não fabricamos pixels via resize artificial.
                    if sys.platform == "win32":
                        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.preferred_size[0])
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.preferred_size[1])
                    camera.set(cv2.CAP_PROP_FPS, self.preferred_fps)
                    if not self._probe(camera):
                        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        if not self._probe(camera):
                            continue
                    self._camera = camera
                    camera = None
                    return
                finally:
                    if camera is not None:
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
