"""Asynchronous ONNX loading and pluggable neural enhancement strategy."""
from __future__ import annotations

import logging
import threading
from dataclasses import replace
from time import monotonic

import cv2

from .filters import BasicEnhancer
from .inference import OnnxModelSpec, OnnxRuntimeEngine

logger = logging.getLogger(__name__)


class NeuralEnhancementProcessor:
    """Composition of classical fallback and one external ONNX image model."""

    def __init__(self, engine: OnnxRuntimeEngine):
        self.engine = engine
        self.classical = BasicEnhancer()
        self.task = engine.spec.task

    def process(self, frame, settings):
        if self.task == "denoise":
            # Neural denoiser first, then classic illumination. Avoid applying
            # a second heavy spatial denoiser over the neural result.
            restored = self.engine.enhance(frame)
            correction = self.classical.process(restored, replace(settings, denoise=False))
            base = self.classical._manual(frame.copy(), settings)
        else:
            # For low-light models, use classic color/noise preprocessing, but
            # avoid applying another strong gamma lift on top of neural light.
            correction = self.classical.process(frame, replace(settings, mode="off"))
            restored = self.engine.enhance(correction)
            base = correction
            correction = restored
        weight = max(0.0, min(float(settings.strength), 1.0))
        return cv2.addWeighted(base, 1.0 - weight, correction, weight, 0)

    def reset(self):
        self.classical.reset()


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
        if preference not in ("auto", "cpu", "cuda", "directml"):
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
