"""Regression tests: small ONNX denoisers must not erase 720p edges."""
from types import SimpleNamespace

import cv2
import numpy as np

from apps.camera.services.filters import ImageSettings
from apps.camera.services.model_manager import NeuralEnhancementProcessor


class BlurringFakeEngine:
    """Simulate a small neural model whose resized output is less detailed."""

    def __init__(self, low_res=True):
        self.spec = SimpleNamespace(
            task="denoise",
            input_width=32 if low_res else 128,
            input_height=24 if low_res else 96,
        )

    def enhance(self, frame):
        return cv2.GaussianBlur(frame, (0, 0), sigmaX=7)


def test_low_res_ai_keeps_native_resolution_edges():
    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    frame[:, :64] = 30
    frame[:, 64:] = 220
    engine = BlurringFakeEngine(low_res=True)
    result = NeuralEnhancementProcessor(engine).process(
        frame, ImageSettings(mode="off", strength=0.8, denoise=True)
    )
    naive = cv2.addWeighted(frame, 0.2, engine.enhance(frame), 0.8, 0)
    # The center edge should be sharper than global blending.
    assert abs(int(result[48, 63, 0]) - 30) < abs(int(naive[48, 63, 0]) - 30)
    assert abs(int(result[48, 64, 0]) - 220) < abs(int(naive[48, 64, 0]) - 220)
    assert result.shape == frame.shape and result.dtype == np.uint8


def test_neural_denoising_does_not_stack_classical_denoiser():
    frame = np.full((96, 128, 3), 125, dtype=np.uint8)
    processor = NeuralEnhancementProcessor(BlurringFakeEngine())
    observed = []

    def fake_classic(result, settings):
        observed.append(settings.denoise)
        return result

    processor.classical.process = fake_classic
    processor.process(frame, ImageSettings(mode="auto", denoise=True, strength=0.5))
    assert observed == [False]


def test_zero_neural_strength_preserves_frame():
    frame = np.random.default_rng(9).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    result = NeuralEnhancementProcessor(BlurringFakeEngine()).process(
        frame, ImageSettings(mode="off", strength=0)
    )
    np.testing.assert_array_equal(result, frame)
    assert result is not frame
