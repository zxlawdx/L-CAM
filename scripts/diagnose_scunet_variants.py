"""Crash-contained SCUNet/DirectML compatibility matrix.

This script does not use the webcam or modify ONNX files. It executes each
variant in a subprocess to retain the main command after a native AV and
records actual GPU nodes when inference succeeds.

Default plan is deliberately small and sequential, to avoid a flood of driver
crashes: 64x64 with optimizations disabled/basic on adapters 0 and 1. If one
passes, it automatically tests 128 and 256 with that same setting.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from scripts.benchmark_scunet import DEFAULT_MANIFEST
from scripts.diagnose_scunet import exit_description

ROOT = Path(__file__).resolve().parents[1]
OPT_LEVELS = {
    "disable": "ORT_DISABLE_ALL",
    "basic": "ORT_ENABLE_BASIC",
    "extended": "ORT_ENABLE_EXTENDED",
    "all": "ORT_ENABLE_ALL",
}


def describe_graph(manifest: Path) -> dict:
    from apps.camera.services.inference import OnnxModelSpec
    import onnx

    spec = OnnxModelSpec.load(manifest)
    # External tensors are not loaded here: this inspection remains lightweight.
    graph = onnx.load(str(spec.model_path), load_external_data=False)
    return {
        "name": spec.name,
        "sha256": spec.sha256,
        "onnx_opset": [
            {"domain": x.domain, "version": x.version} for x in graph.opset_import
        ],
        "node_count": len(graph.graph.node),
        "operator_count": dict(sorted(Counter(x.op_type for x in graph.graph.node).items())),
        "input_shapes": [
            [str(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param
             for dim in x.type.tensor_type.shape.dim]
            for x in graph.graph.input
        ],
    }


def child(manifest: Path, device: int, size: int, optimization: str) -> int:
    from apps.camera.services.inference import OnnxModelSpec
    import numpy as np
    import onnxruntime as ort

    spec = OnnxModelSpec.load(manifest)
    if spec.layout != "NCHW" or spec.task != "denoise":
        raise ValueError("Este diagnóstico é específico para denoisers SCUNet NCHW.")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = getattr(ort.GraphOptimizationLevel, OPT_LEVELS[optimization])
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_mem_pattern = False
    opts.enable_profiling = True

    # Only reduce shape-dependent resource allocations for this experiment.
    input_dims = spec.input_alignment
    if size % input_dims:
        raise ValueError("Dimensão não alinhada ao modelo.")
    print(
        f"[SCUNet GPU] ORT={ort.__version__} adapter={device} "
        f"tile={size} optimization={optimization}",
        flush=True,
    )
    session = ort.InferenceSession(
        str(spec.model_path),
        sess_options=opts,
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        provider_options=[{"device_id": str(device)}, {}],
    )
    print("[SCUNet GPU] Sessão criada; executando primeiro Run()...", flush=True)
    name = session.get_inputs()[0].name
    input_value = np.random.default_rng(17).random((1, 3, size, size), dtype=np.float32)
    output = session.run(None, {name: input_value})[0]
    if output.shape != input_value.shape or not np.isfinite(output).all():
        raise RuntimeError("Saída SCUNet inválida após inferência.")
    profile_path = Path(session.end_profiling())
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        providers = Counter(
            event.get("args", {}).get("provider")
            for event in profile if event.get("cat") == "Node"
        )
        print(json.dumps({
            "ok": providers["DmlExecutionProvider"] > 0,
            "dml_nodes": providers["DmlExecutionProvider"],
            "cpu_nodes": providers["CPUExecutionProvider"],
            "output_shape": list(output.shape),
            "optimization": optimization,
            "adapter": device,
            "tile": size,
        }, ensure_ascii=False), flush=True)
        return 0 if providers["DmlExecutionProvider"] else 3
    finally:
        profile_path.unlink(missing_ok=True)


def run_variant(manifest: Path, device: int, size: int, optimization: str,
                timeout: int = 90, runner=subprocess.run) -> dict:
    command = [
        sys.executable, "-u", "-X", "faulthandler",
        "-m", "scripts.diagnose_scunet_variants",
        "--child", "--manifest", str(manifest), "--device", str(device),
        "--size", str(size), "--optimization", optimization,
    ]
    result = {"adapter": device, "size": size, "optimization": optimization, "ok": False}
    try:
        run = runner(
            command, cwd=str(ROOT), capture_output=True, text=True, errors="replace",
            timeout=timeout, check=False,
        )
        result.update({
            "exit_code": run.returncode,
            "ok": run.returncode == 0,
            "stdout": run.stdout[-6500:],
            "stderr": run.stderr[-12000:],
        })
        if run.returncode:
            result["error"] = exit_description(run.returncode)
    except subprocess.TimeoutExpired as exc:
        result["exit_code"] = None
        result["error"] = f"Timeout após {timeout}s"
        result["stdout"] = str(exc.stdout or b"")[-6500:]
        result["stderr"] = str(exc.stderr or b"")[-12000:]
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Investigar SCUNet/DirectML por otimização e tamanho")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--devices", type=int, nargs="*", default=[0, 1])
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--optimization", choices=tuple(OPT_LEVELS), default="disable")
    p.add_argument("--timeout", type=int, default=90)
    p.add_argument("--full", action="store_true",
                   help="Inclui Extended e All. Por padrão faz só Disable e Basic.")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.size not in (64, 128, 256) or any(x not in (0, 1, 2, 3) for x in args.devices) or not 15 <= args.timeout <= 240:
        p.error("Tamanho/device/timeout fora dos limites.")
    if args.child:
        raise SystemExit(child(args.manifest, args.devices[0] if args.devices else 0,
                               args.size, args.optimization))

    import onnxruntime as ort
    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "onnxruntime": ort.__version__,
            "available_providers": ort.get_available_providers(),
        },
        "model": describe_graph(args.manifest),
        "checks": [],
    }
    levels = ["disable", "basic"] + (["extended", "all"] if args.full else [])
    for device in dict.fromkeys(args.devices):
        successes = []
        for level in levels:
            print(f"[TESTE] adaptador={device}, {args.size}x{args.size}, otimização={level}", flush=True)
            result = run_variant(args.manifest, device, args.size, level, args.timeout)
            report["checks"].append(result)
            print("[OK]" if result["ok"] else "[FALHA]",
                  result.get("error") or result.get("stdout", "")[-600:], flush=True)
            if not result["ok"] and result.get("stderr"):
                print(result["stderr"][-950:], flush=True)
            if result["ok"]:
                successes.append(level)
                break  # Do not continue hammering a GPU that already passed.
        # Only increase resolution when 64x64 actually executed on GPU.
        if successes:
            for size in (128, 256):
                if size <= args.size:
                    continue
                print(f"[CONFIRMAÇÃO] adaptador={device} {size}x{size}", flush=True)
                result = run_variant(args.manifest, device, size, successes[0], args.timeout)
                report["checks"].append(result)
                print("[OK]" if result["ok"] else "[FALHA]",
                      result.get("error") or result.get("stdout", "")[-600:], flush=True)
                if not result["ok"]:
                    if result.get("stderr"):
                        print(result["stderr"][-950:], flush=True)
                    break
    folder = ROOT / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ("scunet-compatibility-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[RELATÓRIO]", path, flush=True)
    successes = [c for c in report["checks"] if c["ok"]]
    if successes:
        print("[CONCLUSÃO] SCUNet executou ao menos uma variante no DirectML; "
              "compare tamanho/otimização e não ative ainda o modo tempo real.", flush=True)
    else:
        print("[CONCLUSÃO] SCUNet não executou nenhuma variante. "
              "DirectML básico funciona, mas SCUNet continua incompatível nesta configuração.", flush=True)


if __name__ == "__main__":
    main()
