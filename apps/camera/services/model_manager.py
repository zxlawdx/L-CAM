"""Asynchronous ONNX loading and pluggable neural enhancement strategy."""
from __future__ import annotations

import logging
import threading
from dataclasses import replace

import cv2
import numpy as np

from .filters import BasicEnhancer
from .inference import OnnxModelSpec, OnnxRuntimeEngine

logger = logging.getLogger(__name__)


class NeuralEnhancementProcessor:
    """Composition of classical fallback and one external ONNX image model."""

    def __init__(self, engine: OnnxRuntimeEngine):
        self.engine = engine
        self.classical = BasicEnhancer()
        self.task = engine.spec.task

    @staticmethod
    def _blend_denoised(frame, denoised, weight: float, low_resolution: bool):
        """Keep native-size edges when a small neural input is upscaled.

        The model still suppresses low-frequency noise in uniform areas. Hair,
        glasses and lettering retain more of the camera's real luminance detail.
        This is *not* super-resolution and does not recover lost sensor data.
        """
        if weight <= 0:
            return frame.copy()
        if not low_resolution:
            return cv2.addWeighted(frame, 1.0 - weight, denoised, weight, 0)
        if frame.shape != denoised.shape:
            raise ValueError("Resultado neural com resolução diferente da captura.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Blur only the *edge detector* so sensor grain is not mistaken for texture.
        guide = cv2.GaussianBlur(gray, (5, 5), 0)
        dx = cv2.Sobel(guide, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(guide, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.clip((cv2.magnitude(dx, dy) - 18.0) / 70.0, 0.0, 1.0)
        # At strong edges no more than 15% of requested neural mix is used.
        alpha = (weight * (1.0 - 0.85 * edge))[:, :, None]
        mixed = (frame.astype(np.float32) * (1.0 - alpha)
                 + denoised.astype(np.float32) * alpha)
        return np.clip(mixed, 0, 255).astype(np.uint8)

    def process(self, frame, settings):
        weight = max(0.0, min(float(settings.strength), 1.0))
        if self.task == "denoise":
            restored = self.engine.enhance(frame)
            spec = self.engine.spec
            smaller = bool(
                spec.input_width and spec.input_height
                and (spec.input_width < frame.shape[1]
                     or spec.input_height < frame.shape[0])
            )
            preserved = self._blend_denoised(frame, restored, weight, smaller)
            # Work at camera resolution for exposure/sharpness, not 320x180.
            # Never stack the classical spatial+temporal denoiser on this model.
            return self.classical.process(preserved, replace(settings, denoise=False))

        # Low-light models must not receive a second strong gamma lift.
        base = self.classical.process(frame, replace(settings, mode="off"))
        restored = self.engine.enhance(base)
        return cv2.addWeighted(base, 1.0 - weight, restored, weight, 0)

    def reset(self):
        self.classical.reset()

    def fallback(self):
        return BasicEnhancer()


class AsyncModelManager:
    """Load one model in the background; publish a processor only after success."""

    def __init__(self, on_ready, processor_factory=NeuralEnhancementProcessor, engine_factory=OnnxRuntimeEngine):
        self._on_ready = on_ready
        self._processor_factory = processor_factory
        self._engine_factory = engine_factory
        self._lock = threading.RLock()
        self._state = {"state": "off", "name": None, "provider": None, "task": None, "error": None}
        self._generation = 0
        self._closed = False
        self._worker = None

    def load(self, manifest_path: str, preference: str = "auto") -> dict:
        if not isinstance(manifest_path, str) or not 1 <= len(manifest_path) <= 4096:
            raise ValueError("Caminho do manifest inválido.")
        if preference not in ("auto", "cpu", "cuda", "directml", "directml:0", "directml:1"):
            raise ValueError("Provedor inválido.")
        with self._lock:
            if self._closed:
                raise RuntimeError("Gerenciador de modelos encerrado.")
            if self._state["state"] == "loading":
                raise RuntimeError("Aguarde a conclusão do carregamento atual.")
            self._generation += 1
            generation = self._generation
            self._state = {"state": "loading", "name": None, "provider": None, "task": None, "error": None}
        self._worker = threading.Thread(
            target=self._load, args=(manifest_path, preference, generation),
            name="lcam-model-loader", daemon=True
        )
        self._worker.start()
        return self.status()

    def _load(self, path, preference, generation) -> None:
        try:
            spec = OnnxModelSpec.load(path)
            engine = self._engine_factory(spec, preference)
            processor = self._processor_factory(engine)
            with self._lock:
                if self._closed or generation != self._generation:
                    return
                self._on_ready(processor)
                self._state = {
                    "state": "ready",
                    "name": spec.name,
                    "provider": engine.provider,
                    "task": spec.task,
                    "error": None,
                    "license": spec.license,
                    "source": spec.source,
                }
        except Exception as exc:
            logger.exception("Não foi possível carregar o modelo ONNX.")
            with self._lock:
                if not self._closed and generation == self._generation:
                    self._state = {
                        "state": "error", "name": None, "provider": None,
                        "task": None, "error": str(exc)[:700],
                    }

    def report_runtime_error(self, message: str) -> None:
        """Invalidate the selected model after a processing failure."""
        with self._lock:
            self._generation += 1
            self._state = {
                "state": "error", "name": None, "provider": None,
                "task": None, "error": str(message)[:700],
            }

    def unload(self) -> dict:
        with self._lock:
            self._generation += 1
            self._on_ready(BasicEnhancer())
            self._state = {"state": "off", "name": None, "provider": None, "task": None, "error": None}
            return dict(self._state)

    def status(self) -> dict:
        with self._lock:
            return dict(self._state)

    def close(self) -> None:
        with self._lock:
            self._generation += 1
            self._closed = True
