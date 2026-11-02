import numpy as np
import pytest

from apps.camera.services.encoding import JpegPreviewEncoder


def test_jpeg_preview_encoder_roundtrip():
    frame = np.full((32, 48, 3), 123, dtype=np.uint8)
    jpeg = JpegPreviewEncoder(90).encode(frame)
    assert jpeg.startswith(b"\\xff\\xd8") if False else jpeg[:2] == bytes([255, 216])
    assert len(jpeg) > 100


@pytest.mark.parametrize("quality", [0, 101])
def test_invalid_quality(quality):
    with pytest.raises(ValueError):
        JpegPreviewEncoder(quality)
