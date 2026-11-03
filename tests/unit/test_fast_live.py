"""Quality and invariants for the lightweight, no-model webcam pipeline."""
from __future__ import annotations

from dataclasses import asdict

import cv2
import numpy as np
import pytest

from apps.camera.services.fast_live import FastLiveEnhancer, shadow_lut
from apps.camera.services.filters import ImageSettings
from apps.camera.services.profiled import ProfiledEnhancer


def test_shadow_curve_preserves_black_and_overexposed_highlights():
    curve = shadow_lut(85)
    assert curve.shape == (256,)
    assert curve.dtype == np.uint8
    assert curve[0] == 0
    np.testing.assert_array_equal(curve[224:], np.arange(224, 256, dtype=np.uint8))
    assert curve[60] > 60
    assert np.all(np.diff(curve.astype(np.int16)) >= 0)


def test_auto_lifts_dark_subject_without_blowing_bright_room():
    frame = np.full((96, 160, 3), 245, dtype=np.uint8)
    frame[:, :60] = 53
    settings = ImageSettings(mode="auto", strength=0.65, denoise=False, sharpness=0)
    result = FastLiveEnhancer().process(frame, settings)
    assert float(result[:, :60].mean()) > 57.0
    assert abs(float(result[:, 60:].mean()) - 245) <= 2
    assert result.shape == frame.shape


def test_luma_native_detail_is_not_spatially_blurred_by_chroma_denoise():
    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    frame[:, :64] = 22
    frame[:, 64:] = 230
    settings = ImageSettings(mode="off", denoise=True, strength=0.7, sharpness=0.0)
    actual = FastLiveEnhancer().process(frame, settings)
    # No luminance smoothing on either side of the vertical edge.
    original_y = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    changed_y = cv2.cvtColor(actual, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    assert np.abs(changed_y.astype(int) - original_y.astype(int)).max() <= 2
    assert changed_y[40, 63] < 30
    assert changed_y[40, 64] > 220


def test_chroma_noise_in_flat_region_is_reduced():
    rng = np.random.default_rng(13)
    ycc = np.zeros((96, 128, 3), dtype=np.uint8)
    ycc[:, :, 0] = 92
    ycc[:, :, 1] = np.clip(128 + rng.normal(0, 19, (96, 128)), 0, 255)
    ycc[:, :, 2] = np.clip(128 + rng.normal(0, 19, (96, 128)), 0, 255)
    frame = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    result = FastLiveEnhancer().process(
        frame, ImageSettings(mode="off", denoise=True, strength=0.7)
    )
    before = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:, :, 1]
    after = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)[:, :, 1]
    assert after.std() < before.std() * 0.9


def test_moving_colour_object_does_not_trail_previous_frame():
    settings = ImageSettings(mode="off", denoise=True, strength=0.8)
    before = np.full((96, 160, 3), 40, dtype=np.uint8)
    before[15:75, 20:60] = (10, 10, 235)
    moving = np.full_like(before, 40)
    moving[15:75, 100:140] = (10, 10, 235)
    running = FastLiveEnhancer()
    running.process(before, settings)
    after = running.process(moving, settings)
    fresh = FastLiveEnhancer().process(moving, settings)
    # The old object's center must be indistinguishable from a fresh frame.
    np.testing.assert_array_equal(after[30:60, 30:50], fresh[30:60, 30:50])


def test_state_reset_when_mode_changes_or_resolution_changes():
    running = FastLiveEnhancer()
    s = ImageSettings(mode="auto", denoise=True)
    running.process(np.full((64, 72, 3), 30, dtype=np.uint8), s)
    assert running._previous_guide is not None
    running.process(np.full((96, 128, 3), 60, dtype=np.uint8), s)
    assert running._previous_guide.shape == (48, 64)
    running.process(np.full((96, 128, 3), 60, dtype=np.uint8),
                    ImageSettings(mode="off", denoise=False))
    assert running._previous_guide is None


def test_input_is_unchanged_and_non_bgr_is_rejected():
    rng = np.random.default_rng(5)
    frame = rng.integers(0, 255, (52, 88, 3), dtype=np.uint8)
    original = frame.copy()
    result = FastLiveEnhancer().process(frame, ImageSettings(mode="low_light", denoise=True))
    assert result.shape == frame.shape and result.dtype == np.uint8
    np.testing.assert_array_equal(frame, original)
    with pytest.raises(ValueError):
        FastLiveEnhancer().process(frame[:, :, 0], ImageSettings())


def test_negative_manual_brightness_clamps_and_never_inverts_black():
    dark = np.full((32, 32, 3), 30, dtype=np.uint8)
    result = FastLiveEnhancer().process(
        dark, ImageSettings(mode="off", brightness=-60, denoise=False)
    )
    assert result.max() == 0


def test_profile_changes_without_reopening_camera():
    pipeline = ProfiledEnhancer()
    frame = np.full((48, 64, 3), 72, dtype=np.uint8)
    fast = ImageSettings(profile="fast", mode="auto", strength=0.5)
    legacy = ImageSettings(profile="legacy", mode="auto", strength=0.5)
    first = pipeline.process(frame, fast)
    second = pipeline.process(frame, legacy)
    third = pipeline.process(frame, fast)
    assert first.shape == second.shape == third.shape == frame.shape
    assert pipeline._active_profile == "fast"
    np.testing.assert_array_equal(first, third)


def test_invalid_profile_is_rejected_and_roundtrips_settings():
    with pytest.raises(ValueError, match="Perfil"):
        ImageSettings(profile="turbo-unknown")
    assert ImageSettings(**asdict(ImageSettings(profile="legacy"))).profile == "legacy"
