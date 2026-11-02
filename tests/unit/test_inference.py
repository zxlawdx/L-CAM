"""Inference adapter contract tests with a fake ONNX Runtime (no model download)."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest

from apps.camera.services.inference import OnnxModelSpec, OnnxRuntimeEngine
from apps.camera.services.model_manager import AsyncModelManager


class IO:
    name = "image"
    type = "tensor(float)"
    def __init__(self, shape):
        self.shape = shape


class FakeSession:
    def __init__(self, *args, **kwargs):
        self.providers = kwargs["providers"]
    def get_inputs(self):
        return [IO([1, 3, "h", "w"])]
    def get_outputs(self):
        return [IO([1, 3, "h", "w"])]
    def get_providers(self):
        return self.providers
    def run(self, names, inputs):
        return [inputs["image"]]


class FakeRuntime:
    class ExecutionMode:
        ORT_SEQUENTIAL = 0
    class SessionOptions:
        def __init__(self):
            self.enable_mem_pattern = True
            self.execution_mode = None
    @staticmethod
    def get_available_providers():
        return ["CPUExecutionProvider"]
    InferenceSession = FakeSession


def manifest(tmp_path, *, digest=None, task="denoise", layout="NCHW"):
    model = tmp_path / "mock.onnx"
    model.write_bytes(b"ONNX placeholder to test manifest only")
    sha = hashlib.sha256(model.read_bytes()).hexdigest()
    location = tmp_path / "model.json"
    location.write_text(json.dumps({
        "name": "Fake test model",
        "task": task,
        "model_path": "mock.onnx",
        "sha256": digest or sha,
        "license": "TEST-NOT-FOR-DISTRIBUTION",
        "source": "https://example.org/test",
        "layout": layout,
    }), encoding="utf-8")
    return location


def test_verified_manifest_and_input_output_contract(tmp_path):
    spec = OnnxModelSpec.load(manifest(tmp_path))
    engine = OnnxRuntimeEngine(spec, preference="cpu", runtime=FakeRuntime)
    frame = np.full((32, 48, 3), (70, 110, 160), dtype=np.uint8)
    result = engine.enhance(frame)
    assert engine.provider == "CPUExecutionProvider"
    assert result.shape == frame.shape
    assert np.abs(result.astype(int) - frame.astype(int)).max() <= 1


def test_rejects_model_hash_mismatch(tmp_path):
    with pytest.raises(ValueError, match="SHA-256"):
        OnnxModelSpec.load(manifest(tmp_path, digest="a" * 64))


def test_rejects_incompatible_provider(tmp_path):
    spec = OnnxModelSpec.load(manifest(tmp_path))
    with pytest.raises(RuntimeError, match="indisponível"):
        OnnxRuntimeEngine(spec, preference="cuda", runtime=FakeRuntime)


def test_async_loader_publishes_model_after_load(tmp_path):
    ready = []
    class FakeEngine:
        provider = "CPUExecutionProvider"
        def __init__(self, spec, preference):
            self.spec = spec
    manager = AsyncModelManager(
        on_ready=ready.append,
        engine_factory=FakeEngine,
        processor_factory=lambda engine: engine,
    )
    manager.load(str(manifest(tmp_path)), "auto")
    deadline = time.monotonic() + 2
    while manager.status()["state"] == "loading" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager.status()["state"] == "ready"
    assert len(ready) == 1
    assert manager.status()["task"] == "denoise"
    assert manager.unload()["state"] == "off"
    manager.close()
