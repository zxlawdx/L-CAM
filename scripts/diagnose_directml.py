"""Diagnose DirectML with tiny ONNX models independently of SCUNet and Vela.

The tiny test networks run in isolated processes. Native crashes cannot
terminate the parent diagnostic. Requires: python -m pip install onnx
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_models(folder: Path) -> dict[str, Path]:
    try:
        import onnx
        import numpy as np
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:
        raise RuntimeError("Instale o gerador de modelos de teste: python -m pip install onnx") from exc
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
    models = {}
    for name in ("add", "conv"):
        if name == "add":
            weight = numpy_helper.from_array(np.full((1, 3, 1, 1), .25, dtype=np.float32), "weight")
            y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 32, 32])
            nodes = [helper.make_node("Add", ["input", "weight"], ["output"])]
        else:
            weight = numpy_helper.from_array(np.full((4, 3, 3, 3), .01, dtype=np.float32), "weight")
            y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 32, 32])
            nodes = [
                helper.make_node("Conv", ["input", "weight"], ["convolved"], pads=[1, 1, 1, 1]),
                helper.make_node("Relu", ["convolved"], ["output"]),
            ]
        graph = helper.make_graph(nodes, "lcam_directml_" + name, [x], [y], [weight])
        model = helper.make_model(graph, producer_name="L-CAM diagnostic",
            opset_imports=[helper.make_operatorsetid("", 17)], ir_version=9)
        onnx.checker.check_model(model)
        dest = folder / ("minimal_" + name + ".onnx")
        onnx.save_model(model, str(dest))
        models[name] = dest
    return models


def describe_returncode(returncode: int | None) -> str:
    if returncode is None:
        return "Tempo limite excedido"
    status = returncode & 0xFFFFFFFF
    known = {
        0xC0000005: "Falha nativa de acesso à memória (0xC0000005)",
        0xC0000017: "Memória insuficiente (0xC0000017)",
        0xC0000135: "DLL nativa ausente (0xC0000135)",
        0xC0000142: "Falha inicializando DLL nativa (0xC0000142)",
    }
    return known.get(status, f"Exit code {returncode} (0x{status:08X})")


def probe(model: Path, provider: str, device: int, timeout: int = 45, runner=subprocess.run) -> dict:
    cmd = [sys.executable, "-u", "-X", "faulthandler", "-m", "scripts.diagnose_directml",
           "--child", "--model", str(model), "--provider", provider, "--device", str(device)]
    result = {"model": model.name, "provider": provider, "device_id": device if provider == "directml" else None}
    try:
        proc = runner(cmd, cwd=str(ROOT), capture_output=True, text=True,
                      errors="replace", timeout=timeout, check=False)
        result.update(ok=proc.returncode == 0, returncode=proc.returncode,
                      stdout=proc.stdout[-13000:], stderr=proc.stderr[-15000:])
        if proc.returncode:
            result["error"] = describe_returncode(proc.returncode)
    except subprocess.TimeoutExpired as exc:
        result.update(ok=False, returncode=None, error=f"Timeout de {timeout}s",
                      stdout=str(exc.stdout or b"")[-13000:], stderr=str(exc.stderr or b"")[-15000:])
    return result


def run_child(model: Path, provider: str, device: int) -> int:
    import numpy as np
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_mem_pattern = False
    opts.enable_profiling = True
    providers = ["CPUExecutionProvider"]
    configs = [{}]
    if provider == "directml":
        providers.insert(0, "DmlExecutionProvider")
        configs.insert(0, {"device_id": str(device)})
    print(f"[WORKER] ORT {ort.__version__} | {model.name} | {providers} | GPU={device}", flush=True)
    session = ort.InferenceSession(str(model), sess_options=opts,
                                   providers=providers, provider_options=configs)
    print("[WORKER] Sessão criada. Iniciando Run()...", flush=True)
    output = session.run(None, {"input": np.full((1, 3, 32, 32), 0.5, dtype=np.float32)})
    if not output or not np.isfinite(output[0]).all():
        raise RuntimeError("ONNX retornou saída inválida.")
    profile = Path(session.end_profiling())
    try:
        events = json.loads(profile.read_text(encoding="utf-8"))
        node_providers = [
            str(event.get("args", {}).get("provider", ""))
            for event in events if event.get("cat") == "Node"
        ]
        nodes_dml = sum(p == "DmlExecutionProvider" for p in node_providers)
        nodes_cpu = sum(p == "CPUExecutionProvider" for p in node_providers)
        print(json.dumps({
            "ok": True, "session_providers": session.get_providers(),
            "gpu_nodes": nodes_dml, "cpu_nodes": nodes_cpu,
            "output_shape": list(output[0].shape),
        }, ensure_ascii=False), flush=True)
        if provider == "directml" and nodes_dml == 0:
            print("[FALHA] Nenhum nó confirmado no DirectML; possível fallback CPU.", flush=True)
            return 3
        return 0
    finally:
        profile.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico de GPU DirectML independente do SCUNet")
    parser.add_argument("--devices", type=int, nargs="*", default=[0, 1])
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--provider", choices=["cpu", "directml"], default="cpu", help=argparse.SUPPRESS)
    parser.add_argument("--device", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        if not args.model:
            parser.error("--model obrigatório no subprocesso")
        raise SystemExit(run_child(args.model, args.provider, args.device))
    if not 5 <= args.timeout <= 180 or any(device not in (0, 1, 2, 3) for device in args.devices):
        parser.error("Timeout ou índice de adaptador inválido.")
    report = {"environment": {"windows": platform.platform(),
                               "python": platform.python_version()}, "checks": []}
    try:
        import onnxruntime as ort
        report["environment"]["onnxruntime"] = ort.__version__
        report["environment"]["providers"] = ort.get_available_providers()
    except ImportError as exc:
        parser.error("onnxruntime não encontrado: " + str(exc))
    with tempfile.TemporaryDirectory(prefix="lcam-minimal-onnx-") as directory:
        for name, model in make_models(Path(directory)).items():
            tasks = [("cpu", 0)] + [("directml", idx) for idx in dict.fromkeys(args.devices)]
            for provider, device in tasks:
                print(f"[TESTE] {name}: {provider} {device if provider == 'directml' else ''}", flush=True)
                result = probe(model, provider, device, args.timeout)
                report["checks"].append(result)
                print("[OK]" if result["ok"] else "[FALHA]",
                      result.get("error", "") or result.get("stdout", "")[-650:], flush=True)
                if not result["ok"]:
                    print(result.get("stderr", "")[-1000:], flush=True)
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    destination = logs / ("directml-minimal-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".json")
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[RELATÓRIO]", destination, flush=True)
    working = [r for r in report["checks"] if r["provider"] == "directml" and r["ok"]]
    print(
        "[CONCLUSÃO] DirectML executou uma rede pequena; investigue a compatibilidade do SCUNet."
        if working else
        "[CONCLUSÃO] Até uma rede pequena falhou; investigue DirectML/driver/ONNX Runtime.",
        flush=True
    )


if __name__ == "__main__":
    main()
