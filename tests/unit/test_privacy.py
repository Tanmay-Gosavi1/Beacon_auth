import numpy as np
import pytest

from beacon import privacy
from beacon.privacy import BlurredFrame, blur, require_blurred, sharpness


def test_blur_preserves_shape_and_dtype(sharp_frame):
    b = blur(sharp_frame, 31, 6.0)
    assert b.image.shape == sharp_frame.shape
    assert b.image.dtype == sharp_frame.dtype


def test_blur_changes_image_and_leaves_input_untouched(sharp_frame):
    before = sharp_frame.copy()
    b = blur(sharp_frame, 31, 6.0)
    assert not np.array_equal(b.image, sharp_frame)
    assert np.array_equal(sharp_frame, before)
    assert not np.shares_memory(b.image, sharp_frame)


def test_larger_sigma_is_less_sharp(sharp_frame):
    values = [sharpness(blur(sharp_frame, 31, s).image) for s in (1.0, 2.0, 4.0, 8.0)]
    assert sharpness(sharp_frame) > values[0]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_blurred_frame_is_read_only(sharp_frame):
    b = blur(sharp_frame, 5, 1.0)
    with pytest.raises(ValueError):
        b.image[0, 0, 0] = 1


def test_blurred_frame_cannot_be_forged(sharp_frame):
    with pytest.raises(TypeError):
        BlurredFrame(image=sharp_frame, kernel=5, sigma=1.0)


def test_require_blurred_rejects_raw_arrays(sharp_frame):
    with pytest.raises(TypeError, match="privacy boundary"):
        require_blurred(sharp_frame)
    assert require_blurred(blur(sharp_frame, 5, 1.0)) is not None


@pytest.mark.parametrize("kernel,sigma", [(4, 1.0), (1, 1.0), (5, 0.0), (5, -2.0)])
def test_blur_rejects_bad_parameters(sharp_frame, kernel, sigma):
    with pytest.raises(ValueError):
        blur(sharp_frame, kernel, sigma)


@pytest.mark.parametrize(
    "bad", [np.zeros((10, 10), np.uint8), np.zeros((10, 10, 3), np.float32), "img"]
)
def test_blur_rejects_bad_images(bad):
    with pytest.raises((TypeError, ValueError)):
        privacy.blur(bad, 5, 1.0)


def test_blur_with_config(sharp_frame, config):
    b = privacy.blur_with(sharp_frame, config.privacy)
    assert (b.kernel, b.sigma) == (config.privacy.blur_kernel, config.privacy.blur_sigma)
