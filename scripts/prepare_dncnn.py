"""Baixa pesos DnCNN originais do autor e exporta ONNX para teste local.

PyTorch so e necessario para converter os pesos; o L-CAM executa o ONNX
pelo adaptador existente. O peso .pth nao sera incluido no repositorio.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

MODEL_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "dncnn_color_blind"
MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_weights(target: Path) -> None:
    if target.is_file():
        if not 100_000 < target.stat().st_size < MAX_DOWNLOAD_BYTES:
            raise ValueError("Peso PTH local tem tamanho inesperado.")
        print(f"Usando peso ja baixado: {target}")
        return
    print(f"Baixando peso original: {MODEL_URL}")
    temporary = target.with_suffix(".part")
    try:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "L-CAM-setup/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as out:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Download de PTH excedeu limite de seguranca.")
                out.write(chunk)
        if temporary.stat().st_size < 100_000:
            raise ValueError("Download PTH incompleto.")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def export_model(weights_path: Path, model_path: Path) -> None:
    import torch
    from torch import nn
    import onnx

    class DnCNN(nn.Module):
        """20 convolucoes, layout de pesos color blind do KAIR, com residual."""

        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            for index in range(20):
                in_channels = 3 if index == 0 else 64
                out_channels = 3 if index == 19 else 64
                layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True))
                if index < 19:
                    layers.append(nn.ReLU(inplace=False))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return x - self.model(x)

    # Acesso limitado a tensores no checkpoint; evita pickle arbitrario.
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("O checkpoint DnCNN nao contem um state_dict.")
    if "state_dict" in state:
        state = state["state_dict"]
    model = DnCNN().eval()
    model.load_state_dict(state, strict=True)
    sample = torch.full((1, 3, 180, 320), 0.25, dtype=torch.float32)
    with torch.no_grad():
        reference = model(sample)
        torch.onnx.export(
            model, sample, str(model_path),
            opset_version=17, input_names=["image"], output_names=["enhanced"],
            dynamic_axes={
                "image": {2: "height", 3: "width"},
                "enhanced": {2: "height", 3: "width"},
            },
            dynamo=False,
        )
    graph = onnx.load(str(model_path))
    onnx.checker.check_model(graph)
    import numpy as np
    import onnxruntime as ort
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: sample.numpy()},
    )[0]
    if not np.allclose(actual, reference.numpy(), atol=1e-4, rtol=1e-4):
        raise RuntimeError("Saida ONNX diferente da rede PyTorch.")
    print("Conversao ONNX validada em CPU contra a rede original.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preparar DnCNN color blind ONNX para L-CAM")
    parser.add_argument("--output", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "dncnn_color_blind.pth"
    output = directory / "dncnn_color_blind.onnx"
    manifest = directory / "dncnn_color_blind.json"
    download_weights(source)
    if not output.exists() or not manifest.exists():
        export_model(source, output)
    data = {
        "name": "DnCNN color blind (KAIR v1.0)",
        "task": "denoise",
        "model_path": output.name,
        "sha256": sha256(output),
        "license": "KAIR code MIT; check separate pretrained weight rights before redistribution",
        "source": MODEL_URL,
        "layout": "NCHW",
        "output_kind": "enhanced",
        "input_width": 320,
        "input_height": 180,
    }
    manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    from apps.camera.services.inference import OnnxModelSpec, OnnxRuntimeEngine
    import numpy as np
    spec = OnnxModelSpec.load(manifest)
    engine = OnnxRuntimeEngine(spec, preference="cpu")
    result = engine.enhance(np.full((180, 320, 3), 50, dtype=np.uint8))
    if result.shape != (180, 320, 3):
        raise RuntimeError("Saida ONNX com formato inesperado.")
    print("Peso original SHA-256:", sha256(source))
    print("ONNX SHA-256:", data["sha256"])
    print("Teste CPU concluido. Manifest local:", manifest)
    print("Teste GPU: python -m scripts.benchmark_onnx", str(manifest), "--provider directml")


if __name__ == "__main__":
    main()
