"""Regression contract for SCUNet-specific Windows DirectML configuration.

A native 0xC0000005 happened on both adapters for the default ONNX session.
User's Windows 11 + ORT 1.24.4 proved that ORT_ENABLE_BASIC with fixed
batch/height/width works on 64/128/256px tiles. Assert production matches it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from apps.camera.services.inference import (
    OnnxRuntimeEngine,
    SCUNET_REAL_PSNR_SHA256,
)


class FakeInput:
    name = "image"
    shape = [1, 3, "height", "width"]
    type = "tensor(float)"


class FakeSessionOptions:
    def __init__(self):
        self.enable_mem_pattern = True
        self.execution_mode = "parallel"
        self.graph_optimization_level = "ALL"
        self.free_overrides = {}

    def add_free_dimension_override_by_name(self, key, value):
        self.free_overrides[key] = value


class FakeSession:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def get_inputs(self):
        return [FakeInput()]

    def get_outputs(self):
        return [FakeInput()]

    def get_providers(self):
        return self.kwargs["providers"]

    def run(self, names, inputs):
        return [inputs["image"]]


class FakeRuntime:
    class GraphOptimizationLevel:
        ORT_ENABLE_BASIC = "BASIC"

    class ExecutionMode:
        ORT_SEQUENTIAL = "sequential"

    SessionOptions = FakeSessionOptions
    InferenceSession = FakeSession

    @staticmethod
    def get_available_providers():
        return ["DmlExecutionProvider", "CPUExecutionProvider"]


class Spec:
    name = "SCUNet test"
    task = "denoise"
    model_path = Path("model.onnx")
    layout = "NCHW"
    output_kind = "enhanced"
    input_width = None
    input_height = None
    tile_size = 256
    tile_overlap = 32
    input_alignment = 8

    def __init__(self, sha256=SCUNET_REAL_PSNR_SHA256):
        self.sha256 = sha256

    def validate(self):
        pass


def test_scunet_directml_uses_proven_basic_optimization_and_static_dims():
    engine = OnnxRuntimeEngine(Spec(), "directml:1", runtime=FakeRuntime)
    opts = engine._session.kwargs["sess_options"]
    assert opts.enable_mem_pattern is False
    assert opts.execution_mode == FakeRuntime.ExecutionMode.ORT_SEQUENTIAL
    assert opts.graph_optimization_level == FakeRuntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    assert opts.free_overrides == {"batch": 1, "height": 256, "width": 256}
    assert engine._session.kwargs["provider_options"][0] == {"device_id": "1"}
    assert engine.provider == "DmlExecutionProvider"
    assert engine.dml_device_id == 1


def test_non_scunet_model_keeps_existing_directml_behavior():
    engine = OnnxRuntimeEngine(Spec(sha256="a" * 64), "directml:0", runtime=FakeRuntime)
    opts = engine._session.kwargs["sess_options"]
    assert opts.graph_optimization_level == "ALL"
    assert opts.free_overrides == {}
    assert opts.enable_mem_pattern is False


def test_scunet_cpu_stays_independent_of_gpu_profile():
    engine = OnnxRuntimeEngine(Spec(), "cpu", runtime=FakeRuntime)
    opts = engine._session.kwargs["sess_options"]
    assert opts.free_overrides == {}
    assert opts.graph_optimization_level == "ALL"
    assert engine.provider == "CPUExecutionProvider"


def test_scunet_tile_engine_retains_native_frame_shape_with_static_input():
    engine = OnnxRuntimeEngine(Spec(), "directml:0", runtime=FakeRuntime)
    raw = np.random.default_rng(11).integers(
        0, 255, size=(359, 641, 3), dtype=np.uint8
    )
    enhanced = engine.enhance(raw)
    assert enhanced.shape == raw.shape
    assert enhanced.dtype == np.uint8
    assert np.abs(enhanced.astype(np.int16) - raw.astype(np.int16)).max() <= 1
