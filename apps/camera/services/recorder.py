"""Video-only fallback recorder. Audio mux is a separate delivery."""
from pathlib import Path
import cv2

class VideoRecorder:
    def __init__(self):
        self._writer = None

    @property
    def recording(self):
        return self._writer is not None

    def start(self, path: str, fps: float, width: int, height: int):
        if self.recording:
            raise RuntimeError("Gravação já iniciada")
        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("Dimensões ou FPS inválidos")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            writer.release()
            raise RuntimeError("Não foi possível criar arquivo MP4")
        self._writer = writer

    def write(self, frame):
        if not self.recording:
            raise RuntimeError("Gravação não iniciada")
        self._writer.write(frame)

    def stop(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
