"""Regression tests for adaptive webcam quality without a physical camera."""
import cv2
import numpy as np
import pytest

from apps.camera.services.filters import BasicEnhancer, ImageSettings


@pytest.fixture()
def enhancer():
    return BasicEnhancer()


def test_dark_image_gets_brighter_in_low_light_mode(enhancer):
    dark = np.full((80, 96, 3), 35, dtype=np.uint8)
    result = enhancer.process(dark, ImageSettings(mode="low_light", strength=0.8))
    assert result.shape == dark.shape
    assert result.dtype == np.uint8
    assert float(np.mean(result)) > float(np.mean(dark)) + 5


def test_auto_mode_does_not_blow_out_bright_scene(enhancer):
    bright = np.full((80, 96, 3), 195, dtype=np.uint8)
    result = enhancer.process(bright, ImageSettings(mode="auto", strength=1))
    assert np.mean(np.abs(result.astype(int) - bright.astype(int))) < 3


def test_auto_mode_lifts_dark_scene(enhancer):
    dark = np.full((64, 64, 3), 40, dtype=np.uint8)
    result = enhancer.process(dark, ImageSettings(mode="auto", strength=0.8))
    assert np.mean(result) > np.mean(dark)


def test_denoise_reduces_random_noise_without_resizing(enhancer):
    rng = np.random.default_rng(42)
    noisy = np.clip(70 + rng.normal(0, 20, (96, 96, 3)), 0, 255).astype(np.uint8)
    result = enhancer.process(noisy, ImageSettings(denoise=True, strength=1))
    assert result.shape == noisy.shape
    assert float(result.var()) < float(noisy.var())


def test_original_input_is_never_mutated(enhancer):
    frame = np.full((64, 64, 3), 80, dtype=np.uint8)
    expected = frame.copy()
    enhancer.process(frame, ImageSettings(mode="low_light", strength=0.9, denoise=True))
    np.testing.assert_array_equal(frame, expected)


def test_off_mode_preserves_pixels_if_manual_settings_neutral(enhancer):
    frame = np.random.default_rng(7).integers(0, 256, (31, 31, 3), dtype=np.uint8)
    result = enhancer.process(frame, ImageSettings(mode="off"))
    np.testing.assert_array_equal(frame, result)
    assert frame is not result


def test_manual_brightness_negative_reduces_intensity_without_absolute_value(enhancer):
    frame = np.full((16, 16, 3), 50, dtype=np.uint8)
    result = enhancer.process(frame, ImageSettings(brightness=-30))
    assert np.all(result == 20)


@pytest.mark.parametrize(
    "settings",
    [
        {"mode": "magic"},
        {"strength": -0.01},
        {"strength": 1.1},
        {"strength": float("nan")},
        {"sharpness": -0.1},
        {"sharpness": 1.2},
        {"sharpness": float("inf")},
        {"brightness": True},
        {"denoise": 1},
    ],
)
def test_invalid_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        ImageSettings(**settings)


def test_invalid_frame_is_rejected(enhancer):
    with pytest.raises(ValueError):
        enhancer.process(np.zeros((32, 32), dtype=np.uint8), ImageSettings())


def test_manual_saturation_does_not_overflow(enhancer):
    frame = np.full((32, 32, 3), (10, 70, 150), dtype=np.uint8)
    result = enhancer.process(frame, ImageSettings(saturation=3))
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255
