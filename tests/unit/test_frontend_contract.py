"""Screen elements used by JS should be present in Vela route template."""
from pathlib import Path


def test_camera_ui_declares_all_new_controls():
    template = (Path(__file__).resolve().parents[2] / "apps/camera/templates/index.html").read_text(encoding="utf-8")
    for name in (
        "mode", "strength", "sharpness", "denoise", "brightness", "contrast",
        "saturation", "reset", "compare", "resolution", "fps", "processing",
        "start", "stop", "record", "finish", "device",
    ):
        assert f'id="lcam-{name}"' in template


def test_template_uses_vela_static_paths():
    template = (Path(__file__).resolve().parents[2] / "apps/camera/templates/index.html").read_text(encoding="utf-8")
    assert "static('camera/css/app.css')" in template
    assert "static('camera/js/app.js')" in template
