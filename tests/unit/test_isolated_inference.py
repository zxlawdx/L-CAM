"""Real subprocess tests: native GPU crash must never kill the Vela process."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apps.camera.services.isolated_inference import IsolatedOnnxRuntimeEngine


class FakeSpec:
    task = "denoise"
    name = "test-model"
    model_path = Path("placeholder.onnx")
    manifest_path = Path("placeholder.json")
    input_width = None
    input_height = None

    def validate(self):
        return None


def child_command(mode):
    return [
        sys.executable, "-u", "-m", "tests.unit.fake_onnx_child",
        "--mode", mode,
    ]


def engine(mode, *, frame_timeout=4):
    return IsolatedOnnxRuntimeEngine(
        FakeSpec(),
        command=child_command(mode),
        startup_timeout=12, frame_timeout=frame_timeout,
    )


def test_isolated_neural_processing_round_trips_pixels_and_keeps_process_alive():
    instance = engine("echo")
    try:
        assert instance.provider == "CPUExecutionProvider"
        frame = np.random.default_rng(12).integers(0, 256, (64, 96, 3), dtype=np.uint8)
        a = instance.enhance(frame)
        b = instance.enhance(frame)
        np.testing.assert_array_equal(a, frame)
        np.testing.assert_array_equal(b, frame)
        assert a is not frame
        assert instance._process.poll() is None
    finally:
        instance.close()
    assert instance._process.poll() is not None


def test_native_gpu_crash_is_caught_without_killing_host_interpreter():
    instance = engine("crash-frame")
    try:
        with pytest.raises(RuntimeError, match="ONNX"):
            instance.enhance(np.zeros((24, 32, 3), dtype=np.uint8))
    finally:
        instance.close()
    # This test continues after the fake subprocess calls os._exit(23).
    assert instance._process.poll() is not None


def test_native_gpu_crash_during_session_creation_is_isolated():
    with pytest.raises(RuntimeError, match="ONNX"):
        engine("crash-init")


def test_worker_timeout_returns_error_and_terminates_stuck_process():
    instance = engine("hang", frame_timeout=1)
    try:
        with pytest.raises(RuntimeError, match="Timeout"):
            instance.enhance(np.zeros((20, 20, 3), dtype=np.uint8))
    finally:
        instance.close()
    assert instance._process.poll() is not None


def test_worker_returns_python_error_without_taking_down_app():
    instance = engine("error")
    try:
        with pytest.raises(RuntimeError, match="Fake ONNX failure"):
            instance.enhance(np.zeros((16, 16, 3), dtype=np.uint8))
    finally:
        instance.close()


def test_worker_malformed_frame_does_not_reach_preview():
    instance = engine("bad-size")
    try:
        with pytest.raises(RuntimeError, match="inválido"):
            instance.enhance(np.zeros((16, 16, 3), dtype=np.uint8))
    finally:
        instance.close()


def test_close_is_idempotent():
    instance = engine("echo")
    instance.close()
    instance.close()
    assert instance._process.poll() is not None
