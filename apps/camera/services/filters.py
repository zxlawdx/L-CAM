"""Estratégias de melhoria de imagem em tempo real, sem dependência do Vela.

O pipeline aplica redução de ruído ANTES da correção de luminosidade:
caso contrário, o ganho de sombra também amplifica ruído cromático.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isfinite

import cv2
import numpy as np

from .temporal import MotionAwareTemporalDenoiser

MODES = frozenset({"off", "auto", "balanced", "low_light"})


@dataclass(frozen=True)
class ImageSettings:
    """Configurações imutáveis para um quadro (sem estado na interface)."""

    brightness: int = 0
    contrast: float = 1.0
    saturation: float = 1.0
    denoise: bool = False
    mode: str = "off"
    strength: float = 0.65
    sharpness: float = 0.0
    profile: str = "fast"

    def __post_init__(self) -> None:
        if type(self.brightness) is not int or not -100 <= self.brightness <= 100:
            raise ValueError("Brilho deve ser inteiro entre -100 e 100.")
        for field, value, minimum, maximum in (
            ("Contraste", self.contrast, 0.0, 3.0),
            ("Saturação", self.saturation, 0.0, 3.0),
            ("Intensidade", self.strength, 0.0, 1.0),
            ("Nitidez", self.sharpness, 0.0, 1.0),
        ):
            if type(value) not in (float, int) or not isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{field} fora do intervalo permitido.")
        if type(self.denoise) is not bool:
            raise ValueError("Redução de ruído deve ser booleana.")
        if self.mode not in MODES:
            raise ValueError("Modo de aprimoramento desconhecido.")
        if self.profile not in ("fast", "legacy"):
            raise ValueError("Perfil de vídeo desconhecido (fast/legacy).")


@lru_cache(maxsize=64)
def _gamma_table(exponent: float) -> np.ndarray:
    """Tabela de consulta de 256 posições, reaproveitada entre quadros."""
    pixels = np.arange(256, dtype=np.float32) / 255.0
    return np.clip(np.power(pixels, exponent) * 255.0, 0, 255).astype(np.uint8)


class BasicEnhancer:
    """Pipeline CPU: crominância -> exposição local -> controles -> nitidez.

    Compatível com a interface FrameProcessor usada por CameraStudio. A classe
    pode ser trocada por um processador ONNX/CUDA sem acoplar a aplicação ao Vela.
    """

    def __init__(self) -> None:
        self._clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
        self._temporal = MotionAwareTemporalDenoiser()
        self._smoothed_mean = None

    def reset(self) -> None:
        """Clear temporal history after camera, mode or scene transitions."""
        self._temporal.reset()
        self._smoothed_mean = None

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Quadro deve ser BGR uint8 com 3 canais.")

    @staticmethod
    def _denoise(ycrcb: np.ndarray, amount: float) -> np.ndarray:
        """Remove ruído de cor, preservando contornos melhor que blur BGR."""
        y, cr, cb = cv2.split(ycrcb)
        clean_y = cv2.bilateralFilter(y, d=5, sigmaColor=22, sigmaSpace=9)
        cr_smooth = cv2.GaussianBlur(cr, (5, 5), 0)
        cb_smooth = cv2.GaussianBlur(cb, (5, 5), 0)
        weight = max(0.0, min(float(amount), 1.0))
        return cv2.merge((
            cv2.addWeighted(y, 1.0 - weight * 0.55, clean_y, weight * 0.55, 0),
            cv2.addWeighted(cr, 1.0 - weight, cr_smooth, weight, 0),
            cv2.addWeighted(cb, 1.0 - weight, cb_smooth, weight, 0),
        ))

    def _enhance_luma(self, ycrcb: np.ndarray, mode: str, strength: float) -> np.ndarray:
        """Corrige principalmente sombras; mantém realces mais protegidos."""
        y, cr, cb = cv2.split(ycrcb)
        current_mean = cv2.mean(y)[0]
        # Smooth automatic exposure estimates so noise/movement does not
        # produce frame-to-frame brightness pumping.
        if self._smoothed_mean is None or abs(current_mean - self._smoothed_mean) > 35:
            self._smoothed_mean = current_mean
        else:
            self._smoothed_mean = 0.88 * self._smoothed_mean + 0.12 * current_mean
        mean = self._smoothed_mean
        if mode == "auto":
            darkness = float(np.clip((135.0 - mean) / 105.0, 0.0, 1.0))
            if darkness < 0.05:
                return ycrcb
            mode = "low_light"
            strength *= darkness

        if mode == "low_light":
            darkness = float(np.clip((150.0 - mean) / 125.0, 0.0, 1.0))
            exponent = round(1.0 - 0.43 * (0.35 + 0.65 * darkness), 2)
            lifted = cv2.LUT(y, _gamma_table(exponent))
            local = self._clahe.apply(lifted)
            # CLAHE em intensidade limitada evita ruído e halos exagerados.
            lifted = cv2.addWeighted(lifted, 0.86, local, 0.14, 0)
            correction = strength
        elif mode == "balanced":
            lifted = self._clahe.apply(y)
            correction = 0.28 * strength
        else:
            return ycrcb

        new_y = cv2.addWeighted(y, 1.0 - correction, lifted, correction, 0)
        return cv2.merge((new_y, cr, cb))

    @staticmethod
    def _manual(frame: np.ndarray, settings: ImageSettings) -> np.ndarray:
        if settings.brightness != 0 or settings.contrast != 1.0:
            adjusted = frame.astype(np.float32) * settings.contrast + settings.brightness
            frame = np.clip(adjusted, 0, 255).astype(np.uint8)
        if settings.saturation != 1.0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            s = hsv[:, :, 1].astype(np.float32) * settings.saturation
            hsv[:, :, 1] = np.clip(s, 0, 255).astype(np.uint8)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return frame

    def process(self, frame: np.ndarray, settings: ImageSettings) -> np.ndarray:
        self._validate_frame(frame)
        if settings.mode == "off" and not settings.denoise and settings.sharpness == 0:
            return self._manual(frame.copy(), settings)

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        if settings.denoise:
            ycrcb = self._temporal.process(ycrcb, strength=settings.strength)
            ycrcb = self._denoise(ycrcb, amount=0.4 + 0.4 * settings.strength)
        else:
            self._temporal.reset()
        if settings.mode != "off":
            ycrcb = self._enhance_luma(ycrcb, settings.mode, settings.strength)

        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        result = self._manual(result, settings)

        if settings.sharpness > 0:
            # Nitidez leve e opcional; em baixa luz, excesso ressuscita o ruído.
            softened = cv2.GaussianBlur(result, (0, 0), sigmaX=0.85)
            amount = 0.55 * settings.sharpness
            result = cv2.addWeighted(result, 1.0 + amount, softened, -amount, 0)
        return result
