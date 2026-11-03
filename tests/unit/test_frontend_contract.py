"""Screen elements used by JS should be present in Vela route template."""
from pathlib import Path


def test_camera_ui_declares_all_new_controls():
    template = (Path(__file__).resolve().parents[2] / "apps/camera/templates/index.html").read_text(encoding="utf-8")
    for name in (
        "mode", "strength", "sharpness", "denoise", "brightness", "contrast",
        "saturation", "reset", "compare", "resolution", "fps", "processing",
        "start", "stop", "record", "finish", "device",
        "microphone", "audio-help", "model-manifest", "model-provider",
        "model-load", "model-unload", "model-status", "preset-select",
        "preset-name", "preset-save", "preset-apply", "preset-delete",
    ):
        assert f'id="lcam-{name}"' in template


def test_template_uses_vela_static_paths():
    template = (Path(__file__).resolve().parents[2] / "apps/camera/templates/index.html").read_text(encoding="utf-8")
    assert "static('camera/css/app.css')" in template
    assert "static('camera/js/app.js')" in template
