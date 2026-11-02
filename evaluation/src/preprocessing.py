"""Image Preprocessing and Blur Layer for Beacon Auth Evaluation.

Manages controlled Gaussian blur levels for robustness experiments and
ensures strict compliance with Beacon's privacy boundary.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from beacon.privacy import BlurredFrame, blur


def auto_kernel(sigma: float) -> int:
    """Computes smallest odd kernel covering +/- 3*sigma to avoid truncating Gaussian."""
    if sigma <= 0.0:
        return 3
    return max(3, 2 * int(math.ceil(3.0 * sigma)) + 1)


class ImagePreprocessor:
    """Applies controlled preprocessing and Gaussian blur to face frames."""

    def __init__(self, default_kernel: int = 7, default_sigma: float = 1.0) -> None:
        self.default_kernel = default_kernel
        self.default_sigma = default_sigma

    def preprocess(
        self,
        image: np.ndarray,
        sigma: Optional[float] = None,
        kernel: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[BlurredFrame]]:
        """Preprocesses an input image according to specified blur parameters.
        
        Args:
            image: BGR uint8 numpy array.
            sigma: Gaussian sigma (defaulting to configured sigma). If 0.0, raw image is returned.
            kernel: Kernel size (odd integer). If None, calculated via auto_kernel.
            
        Returns:
            Tuple of (processed_bgr_array, blurred_frame_object).
        """
        if sigma is None:
            sigma = self.default_sigma
        
        if sigma == 0.0:
            # Baseline unblurred condition: raw image without blur
            # Note: For privacy-strict components, minimal blur or direct backend calls are used.
            return image.copy(), None

        if kernel is None or kernel <= 0:
            kernel = auto_kernel(sigma)
        elif kernel % 2 == 0:
            kernel = kernel + 1

        bframe = blur(image, kernel=kernel, sigma=float(sigma))
        return bframe.image, bframe

