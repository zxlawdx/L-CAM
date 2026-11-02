"""JPEG preview encoder; substituível por streaming nativo futuramente."""
from __future__ import annotations

import cv2
import numpy as np


class JpegPreviewEncoder:
    def __init__(self, quality: int = 86):
        if not 1 <= quality <= 100:
            raise ValueError("Qualidade JPEG deve estar entre 1 e 100.")
        self._params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    def encode(self, frame: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".jpg", frame, self._params)
        if not ok:
            raise RuntimeError("Não foi possível codificar o quadro da prévia.")
        return encoded.tobytes()
