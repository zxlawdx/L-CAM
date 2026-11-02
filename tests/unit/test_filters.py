import numpy as np
import pytest
from apps.camera.services.filters import ImageSettings, BasicEnhancer

def test_default_preserves_black_frame():
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    result = BasicEnhancer().process(frame, ImageSettings())
    assert result.shape == frame.shape
    assert result.dtype == frame.dtype

@pytest.mark.parametrize("kwargs", [{"brightness": 101}, {"contrast": -1}, {"saturation": 4}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        ImageSettings(**kwargs)
