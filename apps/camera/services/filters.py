"""Estratégia simples e substituível de realce de imagem."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class ImageSettings:
    brightness: int = 0
    contrast: float = 1.0
    saturation: float = 1.0
    denoise: bool = False

    def __post_init__(self):
        if not -100 <= self.brightness <= 100:
            raise ValueError("Brilho deve estar entre -100 e 100.")
        if not 0.0 <= self.contrast <= 3.0:
            raise ValueError("Contraste deve estar entre 0 e 3.")
        if not 0.0 <= self.saturation <= 3.0:
            raise ValueError("Saturação deve estar entre 0 e 3.")


class BasicEnhancer:
    def process(self, frame, settings: ImageSettings):
        # clip, ao contrário de convertScaleAbs, preserva brilho negativo.
        adjusted = frame.astype(np.float32) * settings.contrast + settings.brightness
        result = np.clip(adjusted, 0, 255).astype(np.uint8)

        if settings.saturation != 1.0:
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1].astype(np.float32) * settings.saturation
            hsv[:, :, 1] = np.clip(saturation, 0, 255).astype(np.uint8)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if settings.denoise:
            result = cv2.bilateralFilter(result, 5, 35, 35)
        return result
