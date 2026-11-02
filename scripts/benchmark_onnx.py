"""Profile a verified user-supplied ONNX model without requiring a webcam.

Run: python -m scripts.benchmark_onnx "C:\\models\\model.json" --provider auto
"""
from __future__ import annotations

import argparse
import json
import statistics
from time import perf_counter

import numpy as np

from apps.camera.services.inference import OnnxModelSpec, OnnxRuntimeEngine


def run_benchmark(manifest: str, provider: str, frames: int) -> dict:
    started = perf_counter()
    spec = OnnxModelSpec.load(manifest)
    engine = OnnxRuntimeEngine(spec, provider)
    init_ms = (perf_counter() - started) * 1000
    width = spec.input_width or 320
    height = spec.input_height or 180
    # Synthetic frames only, not a measurement of webcam image quality.
    sample = np.full((height, width, 3), 65, dtype=np.uint8)
    timings = []
    engine.enhance(sample)
    for _ in range(frames):
        clock = perf_counter()
        result = engine.enhance(sample)
        timings.append((perf_counter() - clock) * 1000)
        assert result.shape == sample.shape
    sorted_times = sorted(timings)
    return {
        "name": spec.name,
        "task": spec.task,
        "provider_configured": engine.provider,
        "requested_provider": provider,
        "input_resolution": [width, height],
        "initialization_ms": round(init_ms, 1),
        "inference_total_ms_mean": round(statistics.mean(timings), 1),
        "inference_total_ms_p95": round(sorted_times[min(len(timings) - 1, int(frames * 0.95))], 1),
        "theoretical_max_inference_fps": round(1000 / statistics.mean(timings), 1),
        "warning": "Includes conversion and ONNX inference, excludes capture, rendering, recording, and real GPU utilization. Not evidence of webcam FPS.",
    }


def main():
    parser = argparse.ArgumentParser(description="L-CAM local ONNX inference benchmark")
    parser.add_argument("manifest", help="JSON with valid model SHA-256 and tensor contract")
    parser.add_argument("--provider", choices=("auto", "cuda", "directml", "cpu"), default="auto")
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.frames <= 200:
        parser.error("frames deve estar em [1, 200].")
    print(json.dumps(run_benchmark(args.manifest, args.provider, args.frames), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
