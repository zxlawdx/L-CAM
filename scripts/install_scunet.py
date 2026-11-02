"""Install a pre-converted, hash-verified SCUNet Real-PSNR ONNX model.

No PyTorch or pickle execution. Downloads only two published ONNX artifacts.
Never bundle these 77MB weights in git. The Hugging Face model card identifies
Apache-2.0 and links upstream cszn/SCUNet. Hashes are pinned to the file pages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "Heliosoph/scunet-onnx"
BASE_URL = "https://huggingface.co/Heliosoph/scunet-onnx/resolve/main"
MODEL_NAME = "scunet_color_real_psnr.onnx"
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "models" / "scunet_real_psnr"
MAX_ASSET_BYTES = 120 * 1024 * 1024


@dataclass(frozen=True)
class DownloadAsset:
    name: str
    sha256: str
    bytes_limit: int = MAX_ASSET_BYTES

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.name}"


# SHA-256 reported by the Hugging Face file cards; checked on every install.
ASSETS = (
    DownloadAsset(MODEL_NAME, "231be201ab413dbc999d7951caa9844846b93a12a40a41e037d6b5888ed4e88c"),
    DownloadAsset(MODEL_NAME + ".data", "98825ea1210b641c71e5f052f582c70c49fd44b35387ebe2c034268c17df3feb"),
)


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path: Path, asset: DownloadAsset) -> bool:
    return path.is_file() and 0 < path.stat().st_size <= asset.bytes_limit and calculate_sha256(path) == asset.sha256


def download_asset(asset: DownloadAsset, directory: Path, *, opener=None, attempts=3) -> Path:
    """Atomic verified download, reusing a valid cache and deleting bad parts."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / asset.name
    if verified(destination, asset):
        print(f"[OK] Reutilizando {asset.name} (SHA-256 confere).", flush=True)
        return destination
    if destination.exists():
        raise RuntimeError(
            f"Arquivo existente possui SHA-256 inesperado: {destination}. "
            "Renomeie o arquivo manualmente antes de tentar baixar de novo."
        )

    opener = opener or urllib.request.urlopen
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(asset.url, headers={
                "User-Agent": "L-CAM-SCUNet-Installer/1.0",
                "Accept": "application/octet-stream",
            })
            print(f"[{attempt}/{attempts}] Baixando {asset.name}...", flush=True)
            total = 0
            digest = hashlib.sha256()
            with opener(request, timeout=90) as response, partial.open("wb") as output:
                if getattr(response, "status", 200) != 200:
                    raise RuntimeError("Download retornou HTTP diferente de 200.")
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > asset.bytes_limit:
                        raise ValueError("Arquivo excede o limite de tamanho.")
                    digest.update(block)
                    output.write(block)
            if not total or digest.hexdigest() != asset.sha256:
                raise ValueError(f"SHA-256 incorreto para {asset.name}; bytes={total}")
            os.replace(partial, destination)
            print(f"[OK] {asset.name} — {total / 1024**2:.1f} MiB verificados.", flush=True)
            return destination
        except (OSError, urllib.error.URLError, RuntimeError, ValueError) as exc:
            partial.unlink(missing_ok=True)
            if attempt == attempts:
                raise RuntimeError(f"Não foi possível baixar {asset.name}: {exc}") from exc
            print(f"[AVISO] Tentativa falhou: {exc}", flush=True)
            time.sleep(min(attempt * 2, 4))
    raise RuntimeError("Falha inesperada no downloader.")


def install_model(directory: Path, tile_size: int = 384, tile_overlap: int = 32, *, opener=None) -> Path:
    if tile_size not in (256, 320, 384, 448, 512):
        raise ValueError("tile_size deve ser 256, 320, 384, 448 ou 512.")
    if not 16 <= tile_overlap < tile_size // 2 or tile_overlap % 8:
        raise ValueError("tile_overlap inválido; use múltiplo de 8 abaixo de metade do bloco.")
    directory = directory.resolve()
    for asset in ASSETS:
        download_asset(asset, directory, opener=opener)
    info = {
        "name": "SCUNet Real-PSNR (fidelidade, blocos nativos)",
        "task": "denoise",
        "model_path": MODEL_NAME,
        "sha256": ASSETS[0].sha256,
        "external_data": {ASSETS[1].name: ASSETS[1].sha256},
        "license": "Apache-2.0 (Heliosoph SCUNet ONNX; confira avisos e termos upstream)",
        "source": f"https://huggingface.co/{MODEL_ID}",
        "layout": "NCHW",
        "output_kind": "enhanced",
        "input_alignment": 8,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
    }
    path = directory / "scunet_real_psnr.json"
    tmp = path.with_suffix(".json.part")
    try:
        tmp.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

    from apps.camera.services.inference import OnnxModelSpec
    OnnxModelSpec.load(path)
    print(f"[OK] Modelo instalado com manifest verificado: {path}", flush=True)
    print(f"[INFO] Patches {tile_size}x{tile_size} / overlap {tile_overlap}px, sem reduzir 720p a 180p.")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Instala SCUNet Real-PSNR ONNX autenticado para L-CAM")
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--tile-size", type=int, default=384, help="Use 256 para reduzir consumo de VRAM")
    parser.add_argument("--overlap", type=int, default=32)
    args = parser.parse_args()
    try:
        install_model(args.output, args.tile_size, args.overlap)
    except Exception as exc:
        print("[ERRO]", exc, file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
