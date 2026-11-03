"""The A/B benchmarking tool must use one identical frame for both engines."""
from pathlib import Path

import cv2
import numpy as np

from scripts.benchmark_live import benchmark, load_frame


def test_synthetic_benchmark_is_labeled_and_completes():
    frame = load_frame(None, None, width=96, height=64)
    result = benchmark(frame, frames=3, source_label="synthetic")
    assert result["source"] == "synthetic"
    assert set(result["profiles"]) == {"fast", "legacy"}
    assert result["profiles"]["fast"]["mean_ms"] > 0
    assert result["profiles"]["legacy"]["mean_ms"] > 0
    assert result["image_width"] == 96
    assert result["image_height"] == 64
    assert result["speedup_vs_legacy"] > 0
    assert "Não mede webcam" in result["note"]  # scope must remain explicit


def test_ab_exports_original_fast_and_legacy_as_full_size_png(tmp_path):
    frame = np.random.default_rng(4).integers(0, 255, (42, 61, 3), dtype=np.uint8)
    original = frame.copy()
    result = benchmark(frame, frames=2, save=True, destination=tmp_path)
    paths = result["image_files"]
    assert set(paths) == {"original", "fast", "legacy", "comparacao"}
    for path in paths.values():
        assert Path(path).is_file()
    np.testing.assert_array_equal(cv2.imread(paths["original"]), original)
    pair = cv2.imread(paths["comparacao"])
    assert pair.shape == (42, 61 * 3, 3)
    np.testing.assert_array_equal(pair[:, :61], original)
    np.testing.assert_array_equal(pair[:, 61:122], cv2.imread(paths["fast"]))
    np.testing.assert_array_equal(pair[:, 122:], cv2.imread(paths["legacy"]))
    np.testing.assert_array_equal(frame, original)


def test_benchmark_skip_legacy_does_not_report_speedup():
    frame = np.full((32, 48, 3), 30, dtype=np.uint8)
    result = benchmark(frame, frames=2, skip_legacy=True)
    assert set(result["profiles"]) == {"fast"}
    assert "speedup_vs_legacy" not in result
