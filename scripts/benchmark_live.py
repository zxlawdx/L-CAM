"""Reproducible fast-vs-legacy webcam processing benchmark with PNG A/B.

Run in the L-CAM venv. Timings exclude camera acquisition and WebView JPEG.
No weights, DirectML, PyTorch, or Internet required.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from apps.camera.services.capture import CameraCapture
from apps.camera.services.filters import BasicEnhancer, ImageSettings
from apps.camera.services.fast_live import FastLiveEnhancer

OUTPUT = Path.home() / "Pictures" / "L-CAM" / "Comparisons"


def load_frame(path: Path | None, camera_index: int | None,
               width: int, height: int) -> np.ndarray:
    if camera_index is not None:
        camera = CameraCapture(index=camera_index, width=width, height=height)
        try:
            camera.start()
            frame = None
            # Let exposure settle before taking a sample of real noise.
            for _ in range(12):
                frame = camera.read()
            if frame is None:
                raise RuntimeError("Webcam não retornou imagem.")
            return frame.copy()
        finally:
            camera.stop()
    if path is not None:
        file = path.expanduser().resolve()
        if not file.is_file():
            raise FileNotFoundError("Imagem não encontrada: " + str(file))
        frame = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Não é possível abrir a imagem como PNG/JPG.")
        if frame.shape[1] > 3840 or frame.shape[0] > 2160:
            raise ValueError("Imagem grande demais para benchmark.")
        return frame
    # Synthetic dark image for CI/smoke only; never claim camera benchmarks.
    rng = np.random.default_rng(42)
    frame = np.full((height, width, 3), 65, dtype=np.uint8)
    frame[:, width // 2:] = (205, 218, 245)
    noise = rng.normal(0, 12, frame.shape)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def measure(processor, frame: np.ndarray, settings: ImageSettings, frames: int):
    processor.reset()
    processor.process(frame, settings)
    timings = []
    output = None
    for _ in range(frames):
        start = perf_counter()
        output = processor.process(frame, settings)
        timings.append((perf_counter() - start) * 1000.0)
    array = np.asarray(timings, dtype=np.float64)
    return output, {
        "mean_ms": round(float(array.mean()), 2),
        "median_ms": round(float(np.median(array)), 2),
        "p95_ms": round(float(np.percentile(array, 95)), 2),
        "processing_only_fps_estimate": round(float(1000 / array.mean()), 2),
    }


def benchmark(frame, frames=15, mode="auto", strength=0.4, denoise=True,
              sharpness=0.1, skip_legacy=False, save=False,
              destination=OUTPUT, source_label="synthetic") -> dict:
    if not 1 <= frames <= 300:
        raise ValueError("Use 1-300 frames.")
    settings = ImageSettings(
        mode=mode, strength=strength, denoise=denoise,
        sharpness=sharpness,
    )
    report = {
        "source": source_label,
        "image_width": int(frame.shape[1]),
        "image_height": int(frame.shape[0]),
        "frames_per_profile": frames,
        "settings": {
            "mode": mode, "denoise": denoise, "strength": strength,
            "sharpness": sharpness,
        },
        "profiles": {},
        "note": "Somente tempo do filtro OpenCV. Não mede webcam, FPS do sensor, bridge Vela, JPEG, áudio ou gravação.",
    }
    results = {}
    engines = [("fast", FastLiveEnhancer())]
    if not skip_legacy:
        engines.append(("legacy", BasicEnhancer()))
    for label, engine in engines:
        result, stat = measure(engine, frame, replace(settings, profile=label), frames)
        if result.shape != frame.shape or result.dtype != frame.dtype:
            raise RuntimeError("Saída do filtro inválida: " + label)
        results[label] = result
        report["profiles"][label] = stat
    if not skip_legacy:
        fast = report["profiles"]["fast"]["mean_ms"]
        legacy = report["profiles"]["legacy"]["mean_ms"]
        report["speedup_vs_legacy"] = round(legacy / fast, 2) if fast else None

    if save:
        destination = Path(destination).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        label = datetime.now().strftime("%Y%m%d-%H%M%S")
        paths = {"original": destination / f"{label}-original.png"}
        for profile in results:
            paths[profile] = destination / f"{label}-{profile}.png"
        montage = np.concatenate([frame, results["fast"]] + (
            [results["legacy"]] if "legacy" in results else []
        ), axis=1)
        paths["comparacao"] = destination / f"{label}-ab.png"
        for kind, filename in paths.items():
            image = montage if kind == "comparacao" else (
                frame if kind == "original" else results[kind]
            )
            if not cv2.imwrite(str(filename), image):
                raise RuntimeError("Erro salvando PNG: " + str(filename))
        report["image_files"] = {name: str(path) for name, path in paths.items()}
    return report


def main():
    parser = argparse.ArgumentParser(description="Benchmark e imagens comparativas do novo filtro leve")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="Caminho de PNG/JPG original")
    source.add_argument("--camera", type=int, help="Índice da webcam; feche o L-CAM")
    parser.add_argument("--frames", type=int, default=15)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--mode", choices=("auto", "off", "balanced", "low_light"),
                        default="auto")
    parser.add_argument("--strength", type=float, default=0.4)
    parser.add_argument("--sharpness", type=float, default=0.1)
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--save", action="store_true", help="Escreve PNG original/fast/legacy")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not 16 <= args.width <= 3840 or not 16 <= args.height <= 2160:
        parser.error("Resolução fora dos limites.")
    if args.camera is not None and not 0 <= args.camera <= 10:
        parser.error("Índice de câmera inválido.")
    frame = load_frame(args.input, args.camera, args.width, args.height)
    label = "camera" if args.camera is not None else ("file" if args.input else "synthetic")
    result = benchmark(
        frame, frames=args.frames, mode=args.mode, strength=args.strength,
        denoise=not args.no_denoise, sharpness=args.sharpness,
        skip_legacy=args.skip_legacy, save=args.save,
        destination=args.output, source_label=label,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
