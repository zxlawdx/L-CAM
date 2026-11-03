"""Preset repository tests: validation, per-user data and atomic persistence."""
import json
import pytest

from apps.camera.services.filters import ImageSettings
from apps.camera.services.presets import PresetStore


def test_save_load_and_delete(tmp_path):
    store = PresetStore(tmp_path / "my-presets.json")
    assert store.list() == []
    result = store.save("Gravação noturna", ImageSettings(mode="low_light", strength=0.7, denoise=True))
    assert result == ["Gravação noturna"]
    assert store.get("Gravação noturna").mode == "low_light"
    saved = json.loads(store.path.read_text(encoding="utf-8"))
    assert saved["version"] == 1
    assert "Gravação noturna" in saved["presets"]
    assert store.delete("Gravação noturna") == []
    with pytest.raises(ValueError):
        store.get("Gravação noturna")


@pytest.mark.parametrize("name", ["", "../secrets", "bad/name", "\\remote", "x" * 41])
def test_reject_invalid_names(tmp_path, name):
    store = PresetStore(tmp_path / "presets.json")
    with pytest.raises(ValueError):
        store.save(name, ImageSettings())


def test_reject_invalid_preset_file(tmp_path):
    path = tmp_path / "presets.json"
    path.write_text('{"version":1,"presets":{"a":{"brightness":900}}}', encoding="utf-8")
    with pytest.raises(ValueError):
        PresetStore(path).list()


def test_multiple_presets_are_sorted(tmp_path):
    store = PresetStore(tmp_path / "presets.json")
    store.save("Noite", ImageSettings(mode="auto"))
    store.save("Dia", ImageSettings(mode="balanced"))
    assert store.list() == ["Dia", "Noite"]
