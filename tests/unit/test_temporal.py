"""Temporal denoising must preserve motion and reset at scene changes."""
import numpy as np

from apps.camera.services.temporal import MotionAwareTemporalDenoiser


def test_temporal_reduces_static_noise_across_frames():
    rng = np.random.default_rng(21)
    denoiser = MotionAwareTemporalDenoiser()
    previous = None
    original = None
    for _ in range(6):
        frame = np.clip(90 + rng.normal(0, 7, (64, 64, 3)), 0, 255).astype(np.uint8)
        original = frame
        previous = denoiser.process(frame, strength=1.0)
    assert previous.var() < original.var()


def test_scene_cut_does_not_create_ghosting():
    denoiser = MotionAwareTemporalDenoiser()
    dark = np.full((48, 48, 3), 15, dtype=np.uint8)
    bright = np.full_like(dark, 220)
    denoiser.process(dark)
    result = denoiser.process(bright, strength=1.0)
    np.testing.assert_array_equal(result, bright)


def test_reset_drops_old_history():
    denoiser = MotionAwareTemporalDenoiser()
    frame = np.full((20, 30, 3), 100, dtype=np.uint8)
    denoiser.process(frame)
    denoiser.reset()
    second = np.full_like(frame, 150)
    np.testing.assert_array_equal(denoiser.process(second), second)


def test_input_is_never_modified():
    rng = np.random.default_rng(4)
    denoiser = MotionAwareTemporalDenoiser()
    frame = rng.integers(0, 255, (36, 36, 3), dtype=np.uint8)
    original = frame.copy()
    denoiser.process(frame)
    np.testing.assert_array_equal(frame, original)
