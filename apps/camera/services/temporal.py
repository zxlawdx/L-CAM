"""Motion-aware temporal denoising for successive YCrCb webcam frames."""
from __future__ import annotations

import cv2
import numpy as np


class MotionAwareTemporalDenoiser:
    """Blend stationary pixels across time, preserve moving edges.

    A single previous processed frame is used; there is no future-frame
    requirement and thus no algorithmic lookahead delay. On large scene changes,
    temporal history is cleared rather than smearing previous content.
    """

    def __init__(self, motion_threshold: float = 14.0, scene_threshold: float = 36.0):
        self.motion_threshold = float(motion_threshold)
        self.scene_threshold = float(scene_threshold)
        self._previous = None

    def reset(self) -> None:
        self._previous = None

    def process(self, current: np.ndarray, strength: float = 0.7) -> np.ndarray:
        if current.ndim != 3 or current.shape[2] != 3 or current.dtype != np.uint8:
            raise ValueError("Denoiser temporal exige imagem YCrCb uint8.")
        if self._previous is None or self._previous.shape != current.shape:
            self._previous = current.copy()
            return current.copy()

        previous = self._previous
        # Reduce isolated sensor noise before estimating actual motion.
        current_y = cv2.GaussianBlur(current[:, :, 0], (5, 5), 0)
        previous_y = cv2.GaussianBlur(previous[:, :, 0], (5, 5), 0)
        delta = cv2.absdiff(current_y, previous_y)
        if cv2.mean(delta)[0] > self.scene_threshold:
            self.reset()
            self._previous = current.copy()
            return current.copy()

        motion = delta.astype(np.float32)
        # Full blending only in stationary pixels; fade toward current value
        # around moving face/hand edges to avoid ghost trails.
        stable = np.clip((self.motion_threshold * 2.4 - motion) / (self.motion_threshold * 2.4), 0.0, 1.0)
        alpha = 0.58 * float(np.clip(strength, 0.0, 1.0))
        weight = (stable * alpha)[:, :, None]
        out = np.clip(
            current.astype(np.float32) * (1.0 - weight)
            + previous.astype(np.float32) * weight,
            0, 255
        ).astype(np.uint8)
        self._previous = out
        return out
