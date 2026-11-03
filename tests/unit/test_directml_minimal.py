"""DirectML fault probes must not mistake CPU fallback or native crashes for success."""
from pathlib import Path
import subprocess

from scripts.diagnose_directml import describe_returncode, probe


def test_access_violation_has_correct_error_code():
    assert "0xC0000005" in describe_returncode(-1073741819)
    assert "0xC0000005" in describe_returncode(3221225477)


def test_native_gpu_failure_does_not_kill_parent(tmp_path):
    calls = []
    def simulated_crash(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, -1073741819, "", "Windows fatal exception")
    result = probe(tmp_path / "minimal.onnx", "directml", 1, runner=simulated_crash)
    assert not result["ok"]
    assert result["device_id"] == 1
    assert "0xC0000005" in result["error"]
    assert "Windows fatal exception" in result["stderr"]
    assert calls[0][1]["capture_output"] is True
    assert calls[0][1]["check"] is False


def test_probe_recovers_success_and_stderr(tmp_path):
    def simulated_success(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, "GPU nodes: 1", "")
    result = probe(tmp_path / "minimal.onnx", "directml", 0, runner=simulated_success)
    assert result["ok"]
    assert result["returncode"] == 0
    assert "GPU nodes" in result["stdout"]


def test_probe_handles_timeout(tmp_path):
    def simulated_timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 10, output=b"output", stderr=b"timeout")
    result = probe(tmp_path / "minimal.onnx", "cpu", 0, timeout=10, runner=simulated_timeout)
    assert result["ok"] is False
    assert result["returncode"] is None
    assert "Timeout" in result["error"]
