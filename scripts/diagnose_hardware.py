"""Opt-in real webcam benchmark for the user's Windows/Linux machine.

This script is the ONLY place claiming local hardware measurements; it is not
run by CI, as GitHub's runners have no physical webcam or target GTX 1650.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import subprocess
from time import monotonic

import cv2

from apps.camera.services.capture import CameraCapture
from apps.camera.services.filters import BasicEnhancer, ImageSettings
from apps.camera.services.audio import AudioCapture


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))], 2)


def environment() -> dict:
    report = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "ffmpeg_installed": bool(shutil.which("ffmpeg")),
        "ffprobe_installed": bool(shutil.which("ffprobe")),
        "microphones_found": len(AudioCapture.list_devices()),
    }
    try:
        import onnxruntime as ort
        report["onnxruntime"] = ort.__version__
        report["onnx_providers"] = ort.get_available_providers()
    except ImportError:
        report["onnxruntime"] = None
        report["onnx_providers"] = []
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=4, check=False
            )
            if result.returncode == 0:
                report["nvidia_gpus"] = result.stdout.strip().splitlines()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return report


def benchmark(index: int, width: int, height: int, frames: int, mode: str) -> dict:
    camera = CameraCapture(index=index, width=width, height=height)
    processor = BasicEnhancer()
    settings = ImageSettings(mode=mode, denoise=True, strength=0.7, sharpness=0.1)
    capture_ms: list[float] = []
    processing_ms: list[float] = []
    real_resolution = None
    started_at = monotonic()
    try:
        camera.start()
        for _ in range(frames):
            started = monotonic()
            frame = camera.read()
            captured = monotonic()
            processed = processor.process(frame, settings)
            completed = monotonic()
            capture_ms.append(1000 * (captured - started))
            processing_ms.append(1000 * (completed - captured))
            real_resolution = [int(frame.shape[1]), int(frame.shape[0])]
            assert processed.shape == frame.shape
    finally:
        camera.stop()
    elapsed = monotonic() - started_at
    return {
        "camera_index": index,
        "requested_resolution": [width, height],
        "real_resolution": real_resolution,
        "frames": frames,
        "mode": mode,
        "elapsed_seconds": round(elapsed, 2),
        "sequential_pipeline_fps": round(frames / elapsed, 2) if elapsed > 0 else 0.0,
        "capture_ms_median": round(statistics.median(capture_ms), 2),
        "capture_ms_p95": percentile(capture_ms, 0.95),
        "processing_ms_median": round(statistics.median(processing_ms), 2),
        "processing_ms_p95": percentile(processing_ms, 0.95),
        "note": "Benchmark sequencial câmera+filtro; não mede o pipeline assíncrono nem pywebview.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="L-CAM — diagnóstico físico de captura e qualidade.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--mode", choices=("auto", "low_light", "balanced", "off"), default="auto")
    args = parser.parse_args()
    if not 0 <= args.camera <= 10 or not 160 <= args.width <= 3840 or not 120 <= args.height <= 2160 or not 10 <= args.frames <= 2000:
        parser.error("Parâmetros fora dos limites.")
    report = {"environment": environment()}
    try:
        report["benchmark"] = benchmark(args.camera, args.width, args.height, args.frames, args.mode)
    except Exception as exc:
        report["benchmark_error"] = str(exc)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
