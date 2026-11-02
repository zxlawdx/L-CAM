"""Small, versioned and atomic per-user image preset repository."""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from .filters import ImageSettings


def default_preset_path() -> Path:
    if sys.platform == "win32":
        root = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
        return root / "L-CAM" / "presets.json"
    root = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return root / "lcam" / "presets.json"


class PresetStore:
    """Persist only validated image settings; no model paths or credentials."""

    MAX_PRESETS = 30

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else default_preset_path()

    def _read(self) -> dict:
        if not self.path.is_file():
            return {}
        if self.path.stat().st_size > 65536:
            raise ValueError("Arquivo de presets excede o tamanho permitido.")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("Arquivo de presets inválido ou incompatível.")
        presets = data.get("presets")
        if not isinstance(presets, dict):
            raise ValueError("Presets devem ser um objeto JSON.")
        for key, value in presets.items():
            self._validate_name(key)
            if not isinstance(value, dict):
                raise ValueError("Preset inválido.")
            ImageSettings(**value)
        return presets

    @staticmethod
    def _validate_name(name) -> str:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-zÀ-ÿ0-9 _-]{1,40}", name):
            raise ValueError("Nome de preset inválido: use 1–40 letras, números, espaço, _ ou -.")
        return name

    def list(self) -> list[str]:
        return sorted(self._read())

    def get(self, name: str) -> ImageSettings:
        name = self._validate_name(name)
        all_presets = self._read()
        if name not in all_presets:
            raise ValueError("Preset não encontrado.")
        return ImageSettings(**all_presets[name])

    def save(self, name: str, settings: ImageSettings) -> list[str]:
        name = self._validate_name(name)
        if not isinstance(settings, ImageSettings):
            raise ValueError("Configurações de preset inválidas.")
        presets = self._read()
        if name not in presets and len(presets) >= self.MAX_PRESETS:
            raise ValueError("Limite de 30 presets atingido.")
        presets[name] = asdict(settings)
        self._write(presets)
        return sorted(presets)

    def delete(self, name: str) -> list[str]:
        name = self._validate_name(name)
        presets = self._read()
        if name not in presets:
            raise ValueError("Preset não encontrado.")
        del presets[name]
        self._write(presets)
        return sorted(presets)

    def _write(self, presets) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"version": 1, "presets": presets}, ensure_ascii=False, indent=2)
        if len(payload.encode("utf-8")) > 65536:
            raise ValueError("Arquivo de presets excede o tamanho permitido.")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=str(self.path.parent),
                prefix=".lcam-", suffix=".tmp", delete=False
            ) as stream:
                stream.write(payload)
                temporary = Path(stream.name)
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
