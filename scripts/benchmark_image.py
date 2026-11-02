"""Synthetic timing check (not a real-camera FPS guarantee).

Run: python -m scripts.benchmark_image --width 1280 --height 720 --frames 30
"""
from __future__ import annotations

import argparse
import statistics
from time import perf_counter

import numpy as np

from apps.camera.services.filters import BasicEnhancer, ImageSettings, MODES


def main() -> None:
    parser = argparse.ArgumentParser(description="L-CAM image pipeline benchmark")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--mode", choices=sorted(MODES), default="auto")
    args = parser.parse_args()
    if not 16 <= args.width <= 3840 or not 16 <= args.height <= 2160 or not 1 <= args.frames <= 1000:
        parser.error("Resolução ou número de quadros fora dos limites.")

    rng = np.random.default_rng(42)
    # Gradient of dim camera-like pixels and colored noise. No real sensor data.
    gradient = np.linspace(20, 95, args.width, dtype=np.float32)[None, :, None]
    image = np.clip(
        gradient + rng.normal(0, 13, size=(args.height, args.width, 3)),
        0, 255
    ).astype(np.uint8)
    enhancer = BasicEnhancer()
    settings = ImageSettings(mode=args.mode, strength=0.7, denoise=True, sharpness=0.1)

    enhancer.process(image, settings)  # warmup
    timings = []
    for _ in range(args.frames):
        began = perf_counter()
        output = enhancer.process(image, settings)
        timings.append((perf_counter() - began) * 1000.0)
        assert output.shape == image.shape

    average = statistics.mean(timings)
    p95 = sorted(timings)[min(len(timings) - 1, int(len(timings) * 0.95))]
    print(f"{args.width}x{args.height}, {args.frames} frames, modo={args.mode}")
    print(f"Processamento: media={average:.1f}ms, p95={p95:.1f}ms, potencial={1000/average:.1f}fps")
    print("ATENCAO: não inclui captura, Vela bridge, JPEG ou gravação.")
