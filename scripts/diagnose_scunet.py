"""Safely diagnose SCUNet on Windows without crashing the setup process.

Every inference runs in a separate Python subprocess. Native DirectML crashes
can kill the inference child without killing the diagnostic parent. All errors
and exit codes are saved to local logs; no webcam or new model download.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from scripts.benchmark_scunet import DEFAULT_MANIFEST


def exit_description(returncode: int | None) -> str:
    if returncode is None:
        return "Tempo limite excedido"
    unsigned = returncode & 0xFFFFFFFF
    known = {
        0xC0000005: "Falha nativa de acesso à memória (0xC0000005)",
        0xC0000017: "Falha nativa de alocação de memória (0xC0000017)",
        0xC0000409: "Processo nativo abortado (0xC0000409)",
        0xC0000135: "DLL/dependência nativa ausente (0xC0000135)",
        0xC0000142: "Falha ao inicializar DLL nativa (0xC0000142)",
    }
    return known.get(unsigned, f"Processo terminou com código {returncode} (0x{unsigned:08X})")


def probe(
    manifest: Path,
    provider: str,
    *,
    device_id: int = 0,
    tile_size: int = 256,
    timeout: int = 120,
    runner=subprocess.run,
) -> dict:
    args = [
        sys.executable, "-u", "-X", "faulthandler",
        "-m", "scripts.benchmark_scunet",
        "--manifest", str(manifest),
        "--provider", provider,
        "--dml-device-id", str(device_id),
        "--tile-size", str(tile_size),
        "--width", str(tile_size),
        "--height", str(tile_size),
        "--frames", "1",
    ]
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1")
    result = {
        "provider": provider,
        "directml_device_id": device_id if provider == "directml" else None,
        "tile_size": tile_size,
        "command": args,
        "returncode": None,
        "ok": False,
    }
    try:
        proc = runner(
            args, cwd=str(Path(__file__).resolve().parents[1]),
            env=env, capture_output=True, text=True, errors="replace",
            timeout=timeout, check=False,
        )
        result["returncode"] = proc.returncode
        result["stdout"] = proc.stdout[-15000:]
        result["stderr"] = proc.stderr[-20000:]
        result["ok"] = proc.returncode == 0
        if proc.returncode != 0:
            result["explanation"] = exit_description(proc.returncode)
    except subprocess.TimeoutExpired as exc:
        result["explanation"] = f"Timeout de {timeout} segundos. O subprocesso foi encerrado."
        result["stdout"] = str(exc.stdout or b"")[-15000:]
        result["stderr"] = str(exc.stderr or b"")[-20000:]
    except OSError as exc:
        result["explanation"] = str(exc)
    return result


def describe_environment() -> dict:
    values = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "model_manifest": str(DEFAULT_MANIFEST),
    }
    try:
        import onnxruntime as ort
        values["onnxruntime_version"] = ort.__version__
        values["providers"] = ort.get_available_providers()
    except Exception as exc:
        values["runtime_error"] = str(exc)
    if sys.platform == "win32":
        try:
            query = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress"],
                capture_output=True, text=True, timeout=12, check=False,
            )
            if query.returncode == 0:
                values["video_controllers"] = query.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico seguro de SCUNet DirectML")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tile-size", type=int, choices=(256, 320, 384), default=256)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--gpu-devices", type=int, nargs="*", default=[0, 1],
                        help="Índices DirectML a testar separadamente (padrão: 0 e 1)")
    parser.add_argument("--skip-cpu", action="store_true")
    args = parser.parse_args()
    if not 15 <= args.timeout <= 900 or any(i < 0 or i > 8 for i in args.gpu_devices):
        parser.error("Timeout/device_id fora do limite.")

    # Verify graph and externally stored weight hashes before executing anything.
    from apps.camera.services.inference import OnnxModelSpec
    spec = OnnxModelSpec.load(args.manifest)
    print(f"[OK] SHA-256: modelo {spec.name} autenticado.", flush=True)
    report = {"environment": describe_environment(), "checks": []}
    tasks = ([] if args.skip_cpu else [("cpu", 0)]) + [
        ("directml", i) for i in dict.fromkeys(args.gpu_devices)
    ]
    for provider, device in tasks:
        print(f"[TESTE] {provider} / device_id={device} / {args.tile_size}px", flush=True)
        result = probe(
            args.manifest, provider, device_id=device,
            tile_size=args.tile_size, timeout=args.timeout,
        )
        report["checks"].append(result)
        if result["ok"]:
            print("[OK] Inferência concluída.", flush=True)
            print(result.get("stdout", "").strip()[-1300:], flush=True)
        else:
            print("[FALHA]", result.get("explanation", "Erro desconhecido"), flush=True)
            output = result.get("stderr", "") or result.get("stdout", "")
            print(output[-2500:] or "Processo terminou sem mensagens; possível falha nativa.", flush=True)

    folder = Path(__file__).resolve().parents[1] / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    filename = "scunet-diagnostic-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".json"
    dest = folder / filename
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[RELATÓRIO] {dest}", flush=True)
    print("A presença de DmlExecutionProvider não prova que a GTX 1650 executou a rede.", flush=True)
    if not any(x["ok"] and x["provider"] == "directml" for x in report["checks"]):
        print("[AVISO] Nenhuma execução DirectML passou. Não ative SCUNet via GPU no L-CAM ainda.", flush=True)


if __name__ == "__main__":
    main()
