"""Small, explicit Python surface used by the starter's JavaScript."""

from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

from vela.core.bridge import BaseBridge
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STARTER_LINKS = {
    "docs": "https://github.com/zxlawdx/Vela-framework/tree/main/docs",
    "examples": "https://github.com/zxlawdx/Vela-framework/blob/main/docs/API.md",
}


class StarterBridge(BaseBridge):
    """Expose only fixed actions; never accept paths or commands from JS."""

    def increment_counter(self, value: int) -> int:
        """Return an incremented value, rejecting malformed bridge input."""
        if type(value) is not int or not 0 <= value < 1_000_000:
            raise ValueError("O contador deve ser um inteiro entre 0 e 999999.")
        return value + 1

    def starter_action(self, action: str) -> dict[str, object]:
        """Open a fixed reference link or this project's own directory."""
        if not isinstance(action, str) or action not in (*STARTER_LINKS, "project"):
            return {"ok": False, "error": "Ação desconhecida."}
        try:
            if action in STARTER_LINKS:
                if not webbrowser.open(STARTER_LINKS[action], new=2):
                    return {"ok": False, "error": "Não foi possível abrir o navegador."}
            elif sys.platform.startswith("win"):
                os.startfile(str(PROJECT_ROOT))
            else:
                command = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.Popen(
                    [command, str(PROJECT_ROOT)],
                    shell=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            return {"ok": True}
        except (OSError, webbrowser.Error) as exc:
            self.logger.warning(f"Falha na ação do starter {action}: {exc}")
            return {"ok": False, "error": "Não foi possível abrir o recurso neste ambiente."}


from apps.camera.services.studio import CameraStudio


class LCamBridge(StarterBridge):
    """Bridge pública Vela ↔ JS; regras de câmera ficam em CameraStudio."""

    def __init__(self, router=None, config=None, **kwargs):
        super().__init__(router=router, config=config, **kwargs)
        self.studio = CameraStudio()

    def _result(self, operation, *args, **kwargs):
        try:
            return {"ok": True, **operation(*args, **kwargs)}
        except (RuntimeError, ValueError, OSError, TypeError, cv2.error) as exc:
            self.logger.warning(f"L-CAM: {exc}")
            return {"ok": False, "error": str(exc)}

    def camera_start(self, index: int = 0) -> dict:
        return self._result(self.studio.start, index)

    def camera_frame(self, original_preview: bool = False, last_sequence: int = -1) -> dict:
        return self._result(self.studio.frame, original_preview, last_sequence)

    def camera_stop(self) -> dict:
        return self._result(self.studio.stop)

    def camera_status(self) -> dict:
        return self._result(self.studio.status)

    def camera_configure(self, settings: dict) -> dict:
        if not isinstance(settings, dict):
            return {"ok": False, "error": "Configurações inválidas."}
        fields = {"brightness", "contrast", "saturation", "denoise", "mode", "strength", "sharpness"}
        if set(settings) - fields:
            return {"ok": False, "error": "Configuração desconhecida."}
        return self._result(self.studio.configure, **settings)

    def camera_record_start(self, audio_device: int = -1) -> dict:
        if type(audio_device) is not int or audio_device < -1:
            return {"ok": False, "error": "Dispositivo de áudio inválido."}
        return self._result(self.studio.record_start, 30.0, None if audio_device == -1 else audio_device)

    def camera_audio_devices(self) -> dict:
        return self._result(lambda: {"devices": self.studio.audio_devices()})

    def camera_record_status(self) -> dict:
        return self._result(self.studio.recording_status)

    def camera_record_stop(self) -> dict:
        return self._result(self.studio.record_stop)

    def camera_model_load(self, manifest_path: str, provider: str = "auto") -> dict:
        """Start loading a verified local ONNX model; never download weights."""
        return self._result(self.studio.model_load, manifest_path, provider)

    def camera_model_status(self) -> dict:
        return self._result(self.studio.model_status)

    def camera_model_unload(self) -> dict:
        return self._result(self.studio.model_unload)

    def camera_presets_list(self) -> dict:
        return self._result(self.studio.presets_list)

    def camera_presets_save(self, name: str) -> dict:
        return self._result(self.studio.presets_save, name)

    def camera_presets_apply(self, name: str) -> dict:
        return self._result(self.studio.presets_apply, name)

    def camera_presets_delete(self, name: str) -> dict:
        return self._result(self.studio.presets_delete, name)

    def camera_model_catalog(self) -> dict:
        """Report SCUNet installation availability to the Vela UI."""
        return self._result(self.studio.model_catalog)

    def camera_scunet_load(self, provider: str = "auto") -> dict:
        """Activate the locally verified SCUNet model without manual paths."""
        return self._result(self.studio.model_load_default, provider)
