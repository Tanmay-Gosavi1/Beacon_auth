"""Privacy layer: software Gaussian blur and the sharp-frame boundary.

Only a ``BlurredFrame`` may be handed to the identity pipeline (detector,
recognizer, preview). It can be created only by :func:`blur`, so a raw sharp
``ndarray`` cannot reach those components by accident: they check the type and
refuse anything else.

Limit (documented in ARCHITECTURE.md): the sharp frame exists in process memory
before it is blurred. Dropping references lets Python free it but does not
guarantee the bytes are wiped, and nothing here protects against a compromised
process, driver or operating system.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import PrivacyConfig

_TOKEN = object()


@dataclass(frozen=True, eq=False)
class BlurredFrame:
    """A frame that has passed through the privacy blur. Construct with :func:`blur`."""

    image: np.ndarray
    kernel: int
    sigma: float
    _token: object = field(repr=False, default=None)

    def __post_init__(self) -> None:
        if self._token is not _TOKEN:
            raise TypeError("BlurredFrame can only be created by beacon.privacy.blur()")
        self.image.setflags(write=False)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.image.shape


def validate_frame(frame: object) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"expected a numpy image, got {type(frame).__name__}")
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError(f"expected an HxWx3 uint8 BGR image, got {frame.shape} {frame.dtype}")
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        raise ValueError(f"image too small: {frame.shape}")
    return frame


def blur(sharp: np.ndarray, kernel: int, sigma: float) -> BlurredFrame:
    """Return a Gaussian-blurred copy of ``sharp``. The input is not modified."""
    validate_frame(sharp)
    if kernel < 3 or kernel % 2 == 0:
        raise ValueError(f"kernel must be odd and >= 3, got {kernel}")
    if not sigma > 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    out = cv2.GaussianBlur(sharp, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    return BlurredFrame(image=out, kernel=kernel, sigma=float(sigma), _token=_TOKEN)


def blur_with(sharp: np.ndarray, cfg: PrivacyConfig) -> BlurredFrame:
    return blur(sharp, cfg.blur_kernel, cfg.blur_sigma)


def require_blurred(obj: object) -> BlurredFrame:
    """Guard used by every identity-side entry point."""
    if not isinstance(obj, BlurredFrame):
        raise TypeError(
            "privacy boundary violation: identity processing accepts only BlurredFrame, "
            f"got {type(obj).__name__}"
        )
    return obj


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian: a standard focus/sharpness measure (higher = sharper)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
