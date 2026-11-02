"""Benchmark native-resolution SCUNet ONNX without using a physical webcam.

The input is synthetic. This measures only Python preprocessing, ONNX inference
and tile stitching. It does NOT predict full L-CAM preview FPS or picture quality.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np

from apps.camera.services.inference import OnnxModelSpec, OnnxRuntimeEngine

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "models" / "scunet_real_psnr" / "scunet_real_psnr.json"


def benchmark(manifest: str | Path, provider: str, width: int, height: int, frames: int) -> dict:
    spec = OnnxModelSpec.load(manifest)
    setup = perf_counter()
    engine = OnnxRuntimeEngine(spec, preference=provider)
    startup_ms = (perf_counter() - setup) * 1000
    sample = np.random.default_rng(24).integers(
        20, 210, size=(height, width, 3), dtype=np.uint8,
    )
    # One warmup may be slow as a DirectML session compiles shaders.
    start = perf_counter()
    first = engine.enhance(sample)
    warmup_ms = (perf_counter() - start) * 1000
    if first.shape != sample.shape or first.dtype != np.uint8:
        raise RuntimeError("O modelo não preservou o formato e a resolução da câmera.")
    times = []
    for _ in range(frames):
        start = perf_counter()
        result = engine.enhance(sample)
        times.append((perf_counter() - start) * 1000)
        if result.shape != sample.shape:
            raise RuntimeError("Resolução alterada pelo modelo.")
    return {
        "name": spec.name,
        "provider": engine.provider,
        "requested_provider": provider,
        "image_width": width,
        "image_height": height,
        "tile_size": spec.tile_size,
        "tile_overlap": spec.tile_overlap,
        "session_initialization_ms": round(startup_ms, 2),
        "warmup_ms": round(warmup_ms, 2),
        "inference_ms_mean": round(statistics.mean(times), 2),
        "inference_ms_max": round(max(times), 2),
        "inference_only_fps_estimate": round(1000 / statistics.mean(times), 2),
        "note": "Frames sintéticos. Não inclui webcam, JPEG, Vela bridge nem encoder.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Medir SCUNet ONNX por blocos")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--provider", choices=["auto", "cpu", "directml", "cuda"], default="auto")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()
    if not 64 <= args.width <= 1920 or not 64 <= args.height <= 1080 or not 1 <= args.frames <= 60:
        parser.error("Dimensões ou quantidade de quadros fora dos limites.")
    print(json.dumps(
        benchmark(args.manifest, args.provider, args.width, args.height, args.frames),
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
