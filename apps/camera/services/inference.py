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
    external_data: dict[str, str] | None = None
    tile_size: int | None = None
    tile_overlap: int = 32
    input_alignment: int = 1

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
            "external_data", "tile_size", "tile_overlap", "input_alignment",
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
            external_data=data.get("external_data"),
            tile_size=data.get("tile_size"),
            tile_overlap=data.get("tile_overlap", 32),
            input_alignment=data.get("input_alignment", 1),
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
        # ONNX external tensor files must be authenticated too. Validating only
        # the tiny .onnx graph would leave ~73 MB of unverified neural weights.
        if self.external_data is not None:
            if not isinstance(self.external_data, dict) or len(self.external_data) > 8:
                raise ValueError("Lista de dados externos ONNX inválida.")
            for filename, expected in self.external_data.items():
                if (
                    not isinstance(filename, str)
                    or filename in (".", "..")
                    or Path(filename).name != filename
                    or not filename.endswith(".onnx.data")
                    or not isinstance(expected, str)
                    or len(expected) != 64
                    or any(ch not in "0123456789abcdef" for ch in expected.lower())
                ):
                    raise ValueError("Nome/hash de tensor ONNX externo inválido.")
                asset = self.model_path.parent / filename
                if not asset.is_file() or not 0 < asset.stat().st_size <= _MAX_MODEL_BYTES:
                    raise ValueError("Dados externos ONNX ausentes ou fora dos limites: " + filename)
                file_hash = hashlib.sha256()
                with asset.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        file_hash.update(chunk)
                if file_hash.hexdigest() != expected.lower():
                    raise ValueError("SHA-256 do tensor ONNX externo inválido: " + filename)
        if type(self.input_alignment) is not int or self.input_alignment < 1 or self.input_alignment > 64 or self.input_alignment & (self.input_alignment - 1):
            raise ValueError("Alinhamento ONNX deve ser potência de 2 entre 1 e 64.")
        if self.tile_size is not None:
            if type(self.tile_size) is not int or not 64 <= self.tile_size <= 1024 or self.tile_size % self.input_alignment:
                raise ValueError("tile_size deve estar alinhado e entre 64 e 1024.")
            if type(self.tile_overlap) is not int or not 8 <= self.tile_overlap < self.tile_size // 2:
                raise ValueError("tile_overlap deve estar entre 8 e metade do bloco.")
            if self.input_width is not None or self.input_height is not None:
                raise ValueError("Modelos em blocos usam dimensões dinâmicas, não input_width/height.")



class OnnxRuntimeEngine:
    """One ONNX session, reused across all frames and protected by a lock."""

    def __init__(
        self, spec: OnnxModelSpec, preference: str = "auto", runtime=None,
        dml_device_id: int = 0,
    ):
        spec.validate()
        if type(dml_device_id) is not int or not 0 <= dml_device_id <= 8:
            raise ValueError("DirectML device_id deve ser inteiro entre 0 e 8.")
        self.dml_device_id = dml_device_id
        # Allow explicit adapter choice from Vela UI. Adapter enumeration is
        # DirectML/DXGI order, not necessarily the GPU list of nvidia-smi.
        if preference.startswith("directml:"):
            value = preference.partition(":")[2]
            if value not in ("0", "1", "2", "3", "4", "5", "6", "7", "8"):
                raise ValueError("Índice de adaptador DirectML inválido.")
            self.dml_device_id = int(value)
            preference = "directml"
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
                provider_options = [
                    {"device_id": str(self.dml_device_id)} if name == "DmlExecutionProvider" else {}
                    for name in providers
                ]
                session = self._ort.InferenceSession(
                    str(spec.model_path),
                    sess_options=options,
                    providers=providers,
                    provider_options=provider_options,
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
        input_dims = self._session.get_inputs()[0].shape
        self._fixed_height = input_dims[2 if spec.layout == "NCHW" else 1]
        self._fixed_width = input_dims[3 if spec.layout == "NCHW" else 2]
        if spec.input_width is not None and isinstance(self._fixed_width, int) and spec.input_width != self._fixed_width:
            raise ValueError("Largura do manifesto incompatível com o tensor ONNX.")
        if spec.input_height is not None and isinstance(self._fixed_height, int) and spec.input_height != self._fixed_height:
            raise ValueError("Altura do manifesto incompatível com o tensor ONNX.")

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

    def _enhance_single(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Quadro de inferência precisa ser BGR uint8.")
        original_h, original_w = frame.shape[:2]
        target_w = self.spec.input_width or (self._fixed_width if isinstance(self._fixed_width, int) and self._fixed_width > 0 else original_w)
        target_h = self.spec.input_height or (self._fixed_height if isinstance(self._fixed_height, int) and self._fixed_height > 0 else original_h)
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

    @staticmethod
    def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
        if length <= tile:
            return [0]
        stride = tile - 2 * overlap
        result = list(range(0, length - tile + 1, stride))
        if result[-1] != length - tile:
            result.append(length - tile)
        return result

    def _enhance_tiled(self, frame: np.ndarray) -> np.ndarray:
        """Run at camera-native resolution using bounded overlapping tiles.

        No 320x180 downscale. Feathered tile blending avoids visible seams;
        output is cropped back to the exact captured dimensions.
        """
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Quadro de inferência precisa ser BGR uint8.")
        h, w = frame.shape[:2]
        tile = self.spec.tile_size
        overlap = self.spec.tile_overlap
        alignment = self.spec.input_alignment
        H = max(tile, ((h + alignment - 1) // alignment) * alignment)
        W = max(tile, ((w + alignment - 1) // alignment) * alignment)
        pad_mode = "reflect" if min(h, w) > 1 else "edge"
        padded = np.pad(frame, ((0, H - h), (0, W - w), (0, 0)), mode=pad_mode)
        total = np.zeros((H, W, 3), dtype=np.float32)
        weights = np.zeros((H, W, 1), dtype=np.float32)
        xs = self._tile_starts(W, tile, overlap)
        ys = self._tile_starts(H, tile, overlap)

        for y in ys:
            for x in xs:
                region = padded[y:y + tile, x:x + tile]
                denoised = self._enhance_single(region)
                if denoised.shape != region.shape:
                    raise RuntimeError("ONNX retornou resolução inesperada para bloco.")
                wy = np.ones(tile, dtype=np.float32)
                wx = np.ones(tile, dtype=np.float32)
                if y > 0:
                    wy[:overlap] = np.linspace(0.02, 1.0, overlap)
                if y + tile < H:
                    wy[-overlap:] = np.linspace(1.0, 0.02, overlap)
                if x > 0:
                    wx[:overlap] = np.linspace(0.02, 1.0, overlap)
                if x + tile < W:
                    wx[-overlap:] = np.linspace(1.0, 0.02, overlap)
                weight = (wy[:, None] * wx[None, :])[:, :, None]
                total[y:y + tile, x:x + tile] += denoised.astype(np.float32) * weight
                weights[y:y + tile, x:x + tile] += weight

        result = np.clip(total / np.maximum(weights, 1e-7), 0, 255)
        return result[:h, :w].astype(np.uint8)

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        if self.spec.tile_size is not None:
            return self._enhance_tiled(frame)
        return self._enhance_single(frame)
