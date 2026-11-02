"""Composable frame processing strategies."""
from dataclasses import dataclass
import cv2

@dataclass(frozen=True)
class ImageSettings:
    brightness: int = 0
    contrast: float = 1.0
    saturation: float = 1.0
    denoise: bool = False

    def __post_init__(self):
        if not -100 <= self.brightness <= 100:
            raise ValueError("brightness fora de [-100,100]")
        if not 0.0 <= self.contrast <= 3.0:
            raise ValueError("contrast fora de [0,3]")
        if not 0.0 <= self.saturation <= 3.0:
            raise ValueError("saturation fora de [0,3]")

class BasicEnhancer:
    def process(self, frame, settings: ImageSettings):
        result = cv2.convertScaleAbs(frame, alpha=settings.contrast, beta=settings.brightness)
        if settings.saturation != 1.0:
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], settings.saturation)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if settings.denoise:
            result = cv2.bilateralFilter(result, 5, 35, 35)
        return result
