"""SCUNet DirectML variant checks execute in children and preserve native crashes."""
from pathlib import Path
import subprocess

from scripts.diagnose_scunet_variants import run_variant


def test_child_cli_receives_explicit_device_size_and_optimization(tmp_path):
    calls = []
    def runner(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd, 0,
            '{"ok": true, "dml_nodes": 3, "cpu_nodes": 0}',
            "",
        )
    result = run_variant(tmp_path / "m.json", device=1, size=64,
                         optimization="disable", runner=runner)
    assert result["ok"]
    args = calls[0]
    assert args[args.index("--device") + 1] == "1"
    assert args[args.index("--size") + 1] == "64"
    assert args[args.index("--optimization") + 1] == "disable"


def test_native_gpu_failure_is_reported_without_killing_parent(tmp_path):
    def runner(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, -1073741819,
                                           "[SCUNet GPU] Session ready", "Windows fatal exception")
    result = run_variant(tmp_path / "m.json", 0, 64, "basic", runner=runner)
    assert not result["ok"]
    assert "0xC0000005" in result["error"]
    assert result["stderr"] == "Windows fatal exception"


def test_timeout_marks_failure(tmp_path):
    def runner(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 10, output=b"start", stderr=b"hang")
    result = run_variant(tmp_path / "m.json", 1, 64, "disable", timeout=10, runner=runner)
    assert result["ok"] is False
    assert result["exit_code"] is None
    assert "Timeout" in result["error"]


def test_explicit_cpu_fallback_not_marked_as_gpu_success(tmp_path):
    def runner(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 3, '{"dml_nodes": 0, "cpu_nodes": 5}', "")
    result = run_variant(tmp_path / "m.json", 0, 64, "disable", runner=runner)
    assert not result["ok"]


def test_cpu_baseline_can_be_requested_in_separate_child(tmp_path):
    calls = []
    def runner(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "cpu verified", "")
    result = run_variant(tmp_path / "m.json", 0, 64, "disable",
                         runner=runner, provider="cpu")
    assert result["ok"]
    assert result["provider"] == "cpu"
    args = calls[0]
    assert args[args.index("--provider") + 1] == "cpu"
