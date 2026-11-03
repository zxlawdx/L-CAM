"""Camera sensor/driver diagnostic: actual FPS, format, brightness distribution.

Never changes the camera's manual exposure, gain, or white balance settings.
Designed to distinguish low-light 10 FPS at source from slow software filters.
Close L-CAM/OBS before invoking on the same physical webcam.
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from time import monotonic

import cv2
import numpy as np

from apps.camera.services.capture import CameraCapture


def diagnose(camera: CameraCapture, frames: int = 90) -> dict:
    camera.start()
    try:
        info = camera.info()
        timings = []
        read_ms = []
        dark_fraction = []
        highlight_fraction = []
        for _ in range(frames):
            started = monotonic()
            frame = camera.read()
            captured = monotonic()
            timings.append(captured)
            read_ms.append((captured - started) * 1000)
            # Subsample for very cheap brightness diagnostics.
            tiny = cv2.cvtColor(frame[::8, ::8], cv2.COLOR_BGR2GRAY)
            dark_fraction.append(float(np.mean(tiny < 70)))
            highlight_fraction.append(float(np.mean(tiny >= 245)))
        interval = timings[-1] - timings[0] if len(timings) > 1 else 0
        measured_fps = (len(timings) - 1) / interval if interval > 0 else 0
        info["actual_measured_fps"] = round(measured_fps, 2)
        info["frames"] = frames
        info["median_read_ms"] = round(statistics.median(read_ms), 2)
        info["p95_read_ms"] = round(float(np.percentile(read_ms, 95)), 2)
        info["dark_pixels_fraction"] = round(statistics.mean(dark_fraction), 3)
        info["near_white_pixels_fraction"] = round(statistics.mean(highlight_fraction), 3)
        info["notes"] = [
            "FPS medido é do sensor + driver, sem filtros, JPEG, Vela ou gravação.",
            "Valores auto_exposure_raw/exposure_raw/gain_raw não têm interpretação universal.",
            "Atrasos por exposição automática em baixa luz podem reduzir FPS; só testes reais com iluminação diferente podem confirmar.",
            "Para testar impacto da iluminação, repita com uma luz frontal suave (não use flash direto nos olhos).",
        ]
        return info
    finally:
        camera.stop()


def main():
    parser = argparse.ArgumentParser(description="Medir FPS real da webcam e suas configurações negociadas")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=90)
    args = parser.parse_args()
    if not 0 <= args.camera <= 10 or not 16 <= args.width <= 3840 or not 16 <= args.height <= 2160:
        parser.error("Câmera ou resolução fora dos limites.")
    if not 1 <= args.fps <= 120 or not 10 <= args.frames <= 600:
        parser.error("FPS solicitado ou número de quadros fora dos limites.")
    result = diagnose(CameraCapture(
        index=args.camera, width=args.width,
        height=args.height, fps=args.fps,
    ), frames=args.frames)
    folder = Path(__file__).resolve().parents[1] / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ("camera-exposure-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".json")
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Relatório:", path)


if __name__ == "__main__":
    main()
