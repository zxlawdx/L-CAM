"""Contratos entre aplicação e adaptadores de vídeo (Dependency Inversion)."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .filters import ImageSettings


class CameraSource(Protocol):
    index: int

    def start(self) -> None: ...
    def read(self) -> np.ndarray: ...
    def stop(self) -> None: ...


class FrameProcessor(Protocol):
    def process(self, frame: np.ndarray, settings: ImageSettings) -> np.ndarray: ...


class RecordingSink(Protocol):
    @property
    def recording(self) -> bool: ...

    def start(self, path: str, fps: float, width: int, height: int) -> None: ...
    def write(self, frame: np.ndarray) -> None: ...
    def stop(self) -> None: ...


class PreviewEncoder(Protocol):
    def encode(self, frame: np.ndarray) -> bytes: ...
