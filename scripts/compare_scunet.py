"""Produce honest native-resolution A/B PNG comparisons with SCUNet.

This can assess picture quality even if a 4GB GPU cannot run SCUNet at video
framerates. The tool deliberately never modifies the original camera frame.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from apps.camera.services.capture import CameraCapture
from apps.camera.services.filters import ImageSettings
from apps.camera.services.inference import OnnxModelSpec, OnnxRuntimeEngine
from apps.camera.services.model_manager import NeuralEnhancementProcessor
from scripts.benchmark_scunet import DEFAULT_MANIFEST


def capture_still(index: int = 0) -> np.ndarray:
    camera = CameraCapture(index=index, width=1280, height=720)
    try:
        camera.start()
        frame = None
        # Wait for driver exposure to settle. Never use a stale photo.
        for _ in range(12):
            frame = camera.read()
        return frame.copy()
    finally:
        camera.stop()


def compare_frame(
    frame: np.ndarray,
    processor: NeuralEnhancementProcessor,
    *,
    strength: float,
    destination: Path,
    prefix: str = "scunet",
) -> dict:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("A imagem de entrada precisa ser BGR uint8.")
    if frame.shape[1] > 1920 or frame.shape[0] > 1080:
        raise ValueError("O teste está limitado a 1920x1080 para não exceder a VRAM.")
    started = perf_counter()
    # No automatic exposure lift, saturation change or stacked spatial filter.
    enhanced = processor.process(frame, ImageSettings(
        mode="off", denoise=False, strength=strength, sharpness=0.0
    ))
    elapsed = perf_counter() - started
    if enhanced.shape != frame.shape:
        raise RuntimeError("A IA alterou a resolução da câmera.")
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / f"{prefix}-original.png"
    restored_path = destination / f"{prefix}-scunet.png"
    compare_path = destination / f"{prefix}-comparacao.png"
    side_by_side = np.concatenate([frame, enhanced], axis=1)
    for path, image in ((raw_path, frame), (restored_path, enhanced), (compare_path, side_by_side)):
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Não foi possível salvar: {path}")
    return {
        "original": str(raw_path),
        "enhanced": str(restored_path),
        "comparison": str(compare_path),
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "processing_seconds": round(elapsed, 2),
        "strength": strength,
        "provider": processor.engine.provider,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SCUNet: foto original/aprimorada para comparar qualidade")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="Caminho de uma imagem ou vídeo já gravado")
    source.add_argument("--camera", type=int, help="Índice de webcam (feche o L-CAM antes)")
    parser.add_argument("--provider", choices=("auto", "directml", "cuda", "cpu"), default="auto")
    parser.add_argument("--strength", type=float, default=0.4)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=Path.home() / "Pictures" / "L-CAM" / "Comparisons")
    args = parser.parse_args()
    if not 0 <= args.strength <= 1 or not 0 <= (args.camera if args.camera is not None else 0) <= 10:
        parser.error("Força ou dispositivo fora dos limites.")
    if args.camera is not None:
        frame = capture_still(args.camera)
    elif args.input is not None:
        path = args.input.expanduser().resolve()
        if not path.is_file():
            parser.error("Arquivo de imagem/vídeo não encontrado.")
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            video = cv2.VideoCapture(str(path))
            try:
                okay, frame = video.read()
                if not okay or frame is None:
                    raise RuntimeError("Não foi possível ler um quadro deste arquivo.")
            finally:
                video.release()
    else:
        parser.error("Informe --camera 0 ou --input caminho.")
    spec = OnnxModelSpec.load(args.manifest)
    engine = OnnxRuntimeEngine(spec, args.provider)
    processor = NeuralEnhancementProcessor(engine)
    prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(json.dumps(compare_frame(
        frame, processor, strength=args.strength, destination=args.output, prefix=prefix
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
