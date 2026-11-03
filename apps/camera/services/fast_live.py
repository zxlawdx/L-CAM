"""CPU-first webcam enhancement optimized for low-light video, not single photos.

Implemented using OpenCV's native colour transforms/LUTs on 8-bit image data.
Only chroma is downsampled; the full-resolution luminance/detail stays intact.
No neural weights, GPU provider, full-frame bilateral or NLM are required.

The processing recipe is L-CAM original code, informed by the documented
OpenCV techniques and measured performance work of BigCam and IXC Camera.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np


@lru_cache(maxsize=256)
def color_lut(contrast: float, brightness: int) -> np.ndarray:
    """Clamp manual brightness safely; convertScaleAbs reflects negatives."""
    values = np.arange(256, dtype=np.float32)
    return np.clip(np.rint(values * contrast + brightness), 0, 255).astype(np.uint8)


@lru_cache(maxsize=101)
def shadow_lut(level: int) -> np.ndarray:
    """256-entry shadow lift; preserves black and all highlights >= 224."""
    level = max(0, min(100, int(level)))
    values = np.arange(256, dtype=np.float32)
    normalized = values / 255.0
    shifted = np.power(normalized, np.float32(0.62)) * 255.0 - values
    # Smooth highlight roll-off, so bright walls do not become even whiter.
    rolloff = np.clip((224.0 - values) / 112.0, 0.0, 1.0)
    rolloff = rolloff * rolloff * (3.0 - 2.0 * rolloff)
    out = values + shifted * (level / 100.0) * rolloff
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


class FastLiveEnhancer:
    """Low-cost OpenCV enhancement with motion-gated chroma persistence.

    Full-resolution Y has no spatial or temporal smoothing, keeping hair,
    glasses and garment texture. Chroma is treated at half resolution using
    Gaussian filtering and a selective temporal blend in static regions.
    """

    def __init__(self):
        self._previous_guide = None
        self._previous_chroma = None
        self._shadow_percentile = None
        self._last_settings = None
        self._last_shape = None

    def reset(self):
        self._previous_guide = None
        self._previous_chroma = None
        self._shadow_percentile = None
        self._last_settings = None
        self._last_shape = None

    @staticmethod
    def _validate(frame):
        if (not isinstance(frame, np.ndarray) or frame.ndim != 3
                or frame.shape[2] != 3 or frame.dtype != np.uint8):
            raise ValueError("Quadro deve ser BGR uint8.")

    @staticmethod
    def _thumbnail(y):
        h, w = y.shape
        return cv2.resize(y, (min(w, 160), min(h, 90)), interpolation=cv2.INTER_AREA)

    def _chroma_smooth(self, ycrcb: np.ndarray, strength: float):
        h, w = ycrcb.shape[:2]
        size = (max(1, w // 2), max(1, h // 2))
        reduced = cv2.resize(ycrcb, size, interpolation=cv2.INTER_AREA)
        guide = cv2.GaussianBlur(reduced[:, :, 0], (3, 3), 0.75)
        smoothed = cv2.GaussianBlur(reduced[:, :, 1:3], (3, 3), 0.9)
        if self._previous_guide is not None and self._previous_guide.shape == guide.shape:
            change = cv2.absdiff(guide, self._previous_guide)
            if cv2.mean(change)[0] < 30.0:
                # Static pixels receive temporal colour denoising; moving
                # boundaries remain entirely from the current camera frame.
                stable = cv2.compare(change, 9, cv2.CMP_LE)
                blended = cv2.addWeighted(smoothed, 0.72, self._previous_chroma, 0.28, 0)
                cv2.copyTo(blended, stable, smoothed)
        self._previous_guide = guide
        self._previous_chroma = smoothed.copy()
        expanded = cv2.resize(smoothed, (w, h), interpolation=cv2.INTER_LINEAR)
        y, cr, cb = cv2.split(ycrcb)
        cleaned_cr, cleaned_cb = cv2.split(expanded)
        fraction = 0.32 + 0.30 * max(0.0, min(float(strength), 1.0))
        cr = cv2.addWeighted(cr, 1 - fraction, cleaned_cr, fraction, 0)
        cb = cv2.addWeighted(cb, 1 - fraction, cleaned_cb, fraction, 0)
        return y, cr, cb

    def _shadow_lift(self, y: np.ndarray, mode: str, strength: float):
        if mode == "off" or strength <= 0:
            return y
        small = self._thumbnail(y)
        # Look at the darker 15% instead of the global mean: a bright wall
        # must not prevent us from lifting the person in front of it.
        percentile = float(np.percentile(small, 15))
        if self._shadow_percentile is None or abs(percentile - self._shadow_percentile) > 50:
            self._shadow_percentile = percentile
        else:
            # Stable camera exposure with no abrupt per-frame brightness pump.
            self._shadow_percentile = 0.85 * self._shadow_percentile + 0.15 * percentile
        if mode == "balanced":
            amount = 0.30 * strength
        elif mode == "low_light":
            darkness = np.clip((135 - self._shadow_percentile) / 100, 0, 1)
            amount = strength * (0.45 + 0.55 * darkness)
        else:  # auto: only lift when shadows actually need it.
            amount = strength * np.clip((160 - self._shadow_percentile) / 110, 0, 1)
        return cv2.LUT(y, shadow_lut(int(round(amount * 100))))

    def process(self, frame, settings):
        self._validate(frame)
        fingerprint = (settings.mode, settings.denoise)
        if self._last_shape != frame.shape or fingerprint != self._last_settings:
            self.reset()
        self._last_shape = frame.shape
        self._last_settings = fingerprint

        if (settings.mode == "off" and not settings.denoise
                and settings.sharpness == 0 and settings.brightness == 0
                and settings.contrast == 1 and settings.saturation == 1):
            return frame.copy()

        if settings.mode == "off" and not settings.denoise:
            image = frame.copy()
        else:
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            if settings.denoise:
                y, cr, cb = self._chroma_smooth(ycrcb, settings.strength)
            else:
                y, cr, cb = cv2.split(ycrcb)
                self._previous_guide = None
                self._previous_chroma = None
            if settings.mode != "off":
                y = self._shadow_lift(y, settings.mode, settings.strength)
            image = cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)

        if settings.contrast != 1 or settings.brightness:
            image = cv2.LUT(image, color_lut(settings.contrast, settings.brightness))
        if settings.saturation != 1:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            table = np.clip(
                np.rint(np.arange(256) * settings.saturation), 0, 255
            ).astype(np.uint8)
            hsv[:, :, 1] = cv2.LUT(hsv[:, :, 1], table)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if settings.sharpness > 0:
            softened = cv2.GaussianBlur(image, (3, 3), sigmaX=0.65)
            amount = min(0.35, settings.sharpness * 0.35)
            image = cv2.addWeighted(image, 1 + amount, softened, -amount, 0)
        return image
