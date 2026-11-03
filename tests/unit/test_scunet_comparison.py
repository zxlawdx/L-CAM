"""A/B still comparison is useful even if SCUNet runs below real-time FPS."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from scripts.compare_scunet import compare_frame


class FakeProcessor:
    def __init__(self):
        self.engine = SimpleNamespace(provider="TEST-CPU")

    def process(self, frame, settings):
        assert settings.mode == "off"
        assert settings.denoise is False
        assert settings.sharpness == 0
        return np.clip(frame.astype(np.float32) * 0.6 + 40, 0, 255).astype(np.uint8)


def test_comparison_saves_original_processed_and_side_by_side(tmp_path):
    frame = np.random.default_rng(21).integers(0, 255, (57, 91, 3), dtype=np.uint8)
    original = frame.copy()
    result = compare_frame(frame, FakeProcessor(), strength=0.4, destination=tmp_path)
    assert result["width"] == 91
    assert result["height"] == 57
    assert result["provider"] == "TEST-CPU"
    first = cv2.imread(result["original"])
    second = cv2.imread(result["enhanced"])
    pair = cv2.imread(result["comparison"])
    assert first.shape == second.shape == frame.shape
    assert pair.shape == (57, 182, 3)
    np.testing.assert_array_equal(first, original)
    np.testing.assert_array_equal(pair[:, :91], original)
    np.testing.assert_array_equal(pair[:, 91:], second)
    np.testing.assert_array_equal(frame, original)
