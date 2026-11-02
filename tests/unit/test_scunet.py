"""Offline regression tests: source authenticity, ONNX tiling and safe installer.

The CI never downloads the 77 MB model as part of unit tests. ONNX Runtime
integration is exercised separately in an opt-in artifact smoke workflow.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apps.camera.services.inference import OnnxModelSpec, OnnxRuntimeEngine
from scripts import install_scunet


class FakeIO:
    name = "image"
    type = "tensor(float)"
    shape = [1, 3, "height", "width"]


class EchoSession:
    def __init__(self, *args, **kwargs):
        self._providers = kwargs["providers"]
        self.calls = 0

    def get_inputs(self):
        return [FakeIO()]

    def get_outputs(self):
        return [FakeIO()]

    def get_providers(self):
        return self._providers

    def run(self, names, feeds):
        self.calls += 1
        return [feeds["image"]]


class EchoRuntime:
    class SessionOptions:
        def __init__(self):
            self.enable_mem_pattern = True
            self.execution_mode = None

    class ExecutionMode:
        ORT_SEQUENTIAL = 0

    InferenceSession = EchoSession

    @staticmethod
    def get_available_providers():
        return ["CPUExecutionProvider"]


def make_manifest(tmp_path, *, corrupted_data=False):
    model = tmp_path / "scunet.onnx"
    weights = tmp_path / "scunet.onnx.data"
    model.write_bytes(b"model-graph")
    weights.write_bytes(b"weights-file")
    manifest = tmp_path / "scunet.json"
    metadata = {
        "name": "SCUNet fake test",
        "task": "denoise",
        "model_path": model.name,
        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "external_data": {
            weights.name: (
                hashlib.sha256(b"another-weight-file").hexdigest()
                if corrupted_data else hashlib.sha256(weights.read_bytes()).hexdigest()
            )
        },
        "license": "Apache-2.0",
        "source": "https://huggingface.co/Heliosoph/scunet-onnx",
        "layout": "NCHW",
        "output_kind": "enhanced",
        "tile_size": 64,
        "tile_overlap": 16,
        "input_alignment": 8,
    }
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    return manifest


def test_scunet_uses_native_resolution_and_preserves_pixels_on_identity_model(tmp_path):
    spec = OnnxModelSpec.load(make_manifest(tmp_path))
    engine = OnnxRuntimeEngine(spec, preference="cpu", runtime=EchoRuntime)
    camera_image = np.random.default_rng(71).integers(0, 256, (97, 137, 3), dtype=np.uint8)
    output = engine.enhance(camera_image)
    assert output.shape == camera_image.shape
    assert output.dtype == camera_image.dtype
    assert np.max(np.abs(output.astype(int) - camera_image.astype(int))) <= 1
    assert engine._session.calls >= 4


def test_scunet_handles_small_non_aligned_frames_without_downscaling(tmp_path):
    engine = OnnxRuntimeEngine(OnnxModelSpec.load(make_manifest(tmp_path)), "cpu", EchoRuntime)
    frame = np.full((43, 51, 3), (30, 90, 120), dtype=np.uint8)
    output = engine.enhance(frame)
    assert output.shape == (43, 51, 3)
    assert np.max(abs(output.astype(int) - frame.astype(int))) <= 1


def test_scunet_rejects_tampered_external_weights(tmp_path):
    with pytest.raises(ValueError, match="tensor ONNX externo"):
        OnnxModelSpec.load(make_manifest(tmp_path, corrupted_data=True))


@pytest.mark.parametrize("name", ["../outside.onnx.data", "/root/model.onnx.data", "fake.bin"])
def test_scunet_rejects_untrusted_sidecar_paths(tmp_path, name):
    path = make_manifest(tmp_path)
    content = json.loads(path.read_text())
    content["external_data"] = {name: "a" * 64}
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="tensor ONNX externo"):
        OnnxModelSpec.load(path)


@pytest.mark.parametrize("size,overlap", [(73, 16), (64, 0), (64, 33)])
def test_scunet_rejects_invalid_tiling(tmp_path, size, overlap):
    path = make_manifest(tmp_path)
    data = json.loads(path.read_text())
    data["tile_size"] = size
    data["tile_overlap"] = overlap
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        OnnxModelSpec.load(path)


def test_tile_starts_cover_last_edge():
    indices = OnnxRuntimeEngine._tile_starts(1280, 384, 32)
    assert indices[0] == 0
    assert indices[-1] == 896
    assert all(b - a <= 384 for a, b in zip(indices, indices[1:]))


class FakeResponse:
    status = 200

    def __init__(self, data):
        self.data = io.BytesIO(data)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size):
        return self.data.read(size)


def test_installer_checks_exact_sha256_and_reuses_cache(tmp_path):
    content = b"fake weights"
    asset = install_scunet.DownloadAsset("sample.onnx", hashlib.sha256(content).hexdigest())
    hits = []
    def opener(request, timeout):
        hits.append(request.full_url)
        return FakeResponse(content)

    target = install_scunet.download_asset(asset, tmp_path, opener=opener)
    assert target.read_bytes() == content
    target_again = install_scunet.download_asset(asset, tmp_path, opener=opener)
    assert target_again == target
    assert len(hits) == 1


def test_installer_never_keeps_invalid_download(tmp_path):
    asset = install_scunet.DownloadAsset("sample.onnx", "0" * 64)
    with pytest.raises(RuntimeError, match="SHA-256"):
        install_scunet.download_asset(
            asset, tmp_path, opener=lambda req, timeout: FakeResponse(b"bad"),
            attempts=1,
        )
    assert not (tmp_path / "sample.onnx").exists()
    assert not (tmp_path / "sample.onnx.part").exists()


def test_scunet_catalog_does_not_need_model_installed():
    from apps.camera.services.studio import CameraStudio
    studio = CameraStudio()
    result = studio.model_catalog()
    assert result["models"][0]["name"] == "SCUNet Real-PSNR"
    assert isinstance(result["models"][0]["installed"], bool)
