"""DirectML adapter and native crash reporting regressions (no real GPU needed)."""
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from apps.camera.services.inference import OnnxRuntimeEngine
from scripts.diagnose_scunet import exit_description, probe


class FakeIO:
    name = "image"
    type = "tensor(float)"
    shape = [1, 3, "height", "width"]


class FakeSession:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def get_inputs(self):
        return [FakeIO()]

    def get_outputs(self):
        return [FakeIO()]

    def get_providers(self):
        return self.kwargs["providers"]


class FakeORT:
    class SessionOptions:
        def __init__(self):
            self.enable_mem_pattern = True
            self.execution_mode = None

    class ExecutionMode:
        ORT_SEQUENTIAL = "sequential"

    @staticmethod
    def get_available_providers():
        return ["DmlExecutionProvider", "CPUExecutionProvider"]

    InferenceSession = FakeSession


class FakeModelSpec:
    layout = "NCHW"
    input_width = None
    input_height = None
    model_path = Path("fake.onnx")

    def validate(self):
        pass


@pytest.mark.parametrize("device_id", [0, 1])
def test_dml_inference_selects_explicit_adapter(device_id):
    engine = OnnxRuntimeEngine(
        FakeModelSpec(), f"directml:{device_id}", runtime=FakeORT,
    )
    options = engine._session.kwargs["provider_options"]
    assert options[0] == {"device_id": str(device_id)}
    assert engine._session.kwargs["providers"] == [
        "DmlExecutionProvider", "CPUExecutionProvider",
    ]
    opts = engine._session.kwargs["sess_options"]
    assert opts.enable_mem_pattern is False
    assert opts.execution_mode == FakeORT.ExecutionMode.ORT_SEQUENTIAL
    assert engine.dml_device_id == device_id


@pytest.mark.parametrize("preference", ["directml:-1", "directml:999", "directml:x"])
def test_bad_adapter_is_rejected(preference):
    with pytest.raises(ValueError):
        OnnxRuntimeEngine(FakeModelSpec(), preference, runtime=FakeORT)


def test_diag_captures_native_windows_access_violation_without_traceback(tmp_path):
    def fake_runner(args, **kwargs):
        return subprocess.CompletedProcess(args, -1073741819, "[SCUNet] Starting session", "")

    result = probe(
        tmp_path / "test.json", "directml", device_id=1,
        runner=fake_runner,
    )
    assert result["ok"] is False
    assert "0xC0000005" in result["explanation"]
    assert result["directml_device_id"] == 1
    assert result["stdout"].startswith("[SCUNet]")


def test_diag_passes_adapter_and_tile_size_to_benchmark(tmp_path):
    calls = []
    def fake_runner(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "OK", "")

    result = probe(tmp_path / "test.json", "directml", device_id=1,
                   tile_size=256, runner=fake_runner)
    assert result["ok"]
    assert "--dml-device-id" in calls[0]
    assert calls[0][calls[0].index("--dml-device-id") + 1] == "1"
    assert calls[0][calls[0].index("--tile-size") + 1] == "256"
