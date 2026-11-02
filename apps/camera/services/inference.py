"""Optional ONNX Runtime adapter for *user-supplied, licensed* local models.

No neural weights are bundled. This module does not import ONNX Runtime until
an external model is explicitly loaded. CUDA, DirectML and CPU selection is
validated by creating a real session rather than by guessing from GPU names.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)
_MAX_MODEL_BYTES = 400 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class OnnxModelSpec:
    name: str
    task: str
    model_path: Path
    sha256: str
    license: str
    source: str
    layout: str = "NCHW"
    output_kind: str = "enhanced"
    input_width: int | None = None
    input_height: int | None = None

    @classmethod
    def load(cls, manifest: str | Path) -> "OnnxModelSpec":
        file = Path(manifest).expanduser().resolve()
        if not file.is_file() or file.suffix.lower() != ".json":
            raise ValueError("Informe o caminho de um manifest JSON local.")
        if file.stat().st_size > 32768:
            raise ValueError("Manifest do modelo excede 32 KB.")
        data = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Manifest deve conter um objeto JSON.")
        allowed = {
            "name", "task", "model_path", "sha256", "license", "source",
            "layout", "output_kind", "input_width", "input_height",
        }
        if set(data) - allowed:
            raise ValueError("Campos desconhecidos no manifest.")
        for key in ("name", "task", "model_path", "sha256", "license", "source"):
            if not isinstance(data.get(key), str) or not data[key].strip():
                raise ValueError(f"Campo obrigatório: {key}")
        model_file = (file.parent / data["model_path"]).resolve()
        spec = cls(
            name=data["name"][:100],
            task=data["task"],
            model_path=model_file,
            sha256=data["sha256"].lower(),
            license=data["license"][:140],
            source=data["source"][:1024],
            layout=data.get("layout", "NCHW"),
            output_kind=data.get("output_kind", "enhanced"),
            input_width=data.get("input_width"),
            input_height=data.get("input_height"),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.task not in ("denoise", "low_light"):
            raise ValueError("task deve ser denoise ou low_light.")
        if self.layout not in ("NCHW", "NHWC"):
            raise ValueError("layout deve ser NCHW ou NHWC.")
        if self.output_kind not in ("enhanced", "residual"):
            raise ValueError("output_kind deve ser enhanced ou residual.")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("sha256 inválido.")
        if not self.model_path.is_file() or self.model_path.suffix.lower() != ".onnx":
            raise ValueError("Arquivo ONNX não encontrado.")
        if self.model_path.stat().st_size > _MAX_MODEL_BYTES:
            raise ValueError("Modelo excede o limite de segurança de 400 MB.")
        for dimension in (self.input_width, self.input_height):
            if dimension is not None and (type(dimension) is not int or not 32 <= dimension <= 4096):
                raise ValueError("Resolução de inferência inválida.")
        if bool(self.input_width is None) != bool(self.input_height is None):
            raise ValueError("Informe width e height juntos.")
        digest = hashlib.sha256()
        with self.model_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != self.sha256:
            raise ValueError("SHA-256 do ONNX não confere com o manifest.")


class OnnxRuntimeEngine:
    """One ONNX session, reused across all frames and protected by a lock."""

    def __init__(self, spec: OnnxModelSpec, preference: str = "auto", runtime=None):
        spec.validate()
        self.spec = spec
        self._ort = runtime or importlib.import_module("onnxruntime")
        self._session_lock = threading.Lock()
        available = set(self._ort.get_available_providers())
        order = self._provider_order(preference, available)
        self._session = None
        errors = []
        for provider in order:
            try:
                options = self._ort.SessionOptions()
                if provider == "DmlExecutionProvider":
                    options.enable_mem_pattern = False
                    options.execution_mode = self._ort.ExecutionMode.ORT_SEQUENTIAL
                providers = [provider]
                if provider != "CPUExecutionProvider" and "CPUExecutionProvider" in available:
                    providers.append("CPUExecutionProvider")
                session = self._ort.InferenceSession(
                    str(spec.model_path), sess_options=options, providers=providers
                )
                self._check_model_signature(session)
                self._session = session
                self.provider = session.get_providers()[0]
                logger.info("Modelo %s preparado: %s", spec.name, self.provider)
                break
            except Exception as exc:
                logger.warning("Falha inicializando %s para %s: %s", spec.name, provider, exc)
                errors.append(f"{provider}: {exc}")
        if self._session is None:
            raise RuntimeError("Não foi possível carregar ONNX: " + " | ".join(errors)[:1000])
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    @staticmethod
    def _provider_order(preference: str, available: set[str]) -> list[str]:
        values = {
            "cuda": "CUDAExecutionProvider",
            "directml": "DmlExecutionProvider",
            "cpu": "CPUExecutionProvider",
        }
        if preference not in ("auto", *values.keys()):
            raise ValueError("Provedor inválido. Use auto, cuda, directml ou cpu.")
        if preference != "auto":
            selected = values[preference]
            if selected not in available:
                raise RuntimeError(f"Provedor {selected} indisponível neste ambiente.")
            return [selected] + (["CPUExecutionProvider"] if selected != "CPUExecutionProvider" and "CPUExecutionProvider" in available else [])
        ordered = ["CUDAExecutionProvider"]
        if sys.platform == "win32":
            ordered.append("DmlExecutionProvider")
        ordered.append("CPUExecutionProvider")
        return [name for name in ordered if name in available]

    def _check_model_signature(self, session) -> None:
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Neste estágio, apenas modelos ONNX de uma entrada e uma saída de imagem são aceitos.")
        if len(inputs[0].shape) != 4 or len(outputs[0].shape) != 4:
            raise ValueError("Tensor ONNX deve ter 4 dimensões (batch e imagem).")
        channel = 1 if self.spec.layout == "NCHW" else 3
        for io in (inputs[0], outputs[0]):
            if isinstance(io.shape[channel], int) and io.shape[channel] != 3:
                raise ValueError("Modelo ONNX deve receber/retornar 3 canais RGB.")
        if getattr(inputs[0], "type", "tensor(float)") != "tensor(float)":
            raise ValueError("Neste estágio, a entrada ONNX deve ser float32.")

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Quadro de inferência precisa ser BGR uint8.")
        original_h, original_w = frame.shape[:2]
        target_w = self.spec.input_width or original_w
        target_h = self.spec.input_height or original_h
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA) if (target_w, target_h) != (original_w, original_h) else frame
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        if self.spec.layout == "NCHW":
            tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.ascontiguousarray(tensor[None, ...], dtype=np.float32)
        with self._session_lock:
            output = self._session.run([self._output_name], {self._input_name: tensor})[0]
        output = np.asarray(output)
        if output.ndim != 4 or output.shape[0] != 1:
            raise RuntimeError("Formato de saída ONNX inválido.")
        output = output[0]
        if self.spec.layout == "NCHW":
            output = np.transpose(output, (1, 2, 0))
        if output.ndim != 3 or output.shape[2] != 3:
            raise RuntimeError("Modelo ONNX retornou quantidade de canais diferente de 3.")
        if self.spec.output_kind == "residual":
            if output.shape != rgb.shape:
                raise RuntimeError("Saída residual exige dimensão igual à entrada.")
            output = (rgb.astype(np.float32) / 255.0) - output
        if not np.all(np.isfinite(output)):
            raise RuntimeError("Saída ONNX contém NaN/Inf.")
        enhanced = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        if enhanced.shape[:2] != (original_h, original_w):
            enhanced = cv2.resize(enhanced, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
