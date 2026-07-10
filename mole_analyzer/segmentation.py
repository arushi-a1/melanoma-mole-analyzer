"""
Lesion segmentation.

The original prototype identified "mole pixels" with a flat RGB < 100
threshold. That approach fails on any image with shadows, hair, varied
lighting, or skin tone, and produces no actual shape/contour to analyze.

This module segments the lesion from surrounding skin using Otsu's method
on the L*a*b* color space (which separates lightness from color much more
robustly than raw RGB), followed by morphological cleanup, and returns
both a binary mask and the lesion's outer contour.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import SEGMENTATION


class SegmentationError(Exception):
    """Raised when no plausible lesion contour can be found in an image."""


@dataclass
class SegmentationResult:
    mask: np.ndarray          # uint8 binary mask, 255 = lesion
    contour: np.ndarray       # OpenCV contour (largest valid region)
    image_rgb: np.ndarray     # original image as RGB ndarray, for downstream use


def segment_lesion(image_rgb: np.ndarray) -> SegmentationResult:
    """
    Segment a lesion from a cropped RGB skin-lesion image.

    Args:
        image_rgb: HxWx3 uint8 array in RGB order (e.g. from PIL.Image
            converted via np.array(image.convert("RGB"))).

    Returns:
        SegmentationResult with a binary mask and the lesion contour.

    Raises:
        SegmentationError: if no contour of sufficient area is found.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an HxWx3 RGB image array.")

    k = SEGMENTATION.blur_kernel_size | 1  # force odd
    blurred = cv2.GaussianBlur(image_rgb, (k, k), 0)

    lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]

    # Lesions are typically darker than surrounding skin, so Otsu's
    # threshold on the L channel with an inverted binary works well
    # across a wide range of skin tones/lighting.
    _, mask = cv2.threshold(
        l_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Morphological cleanup: close small gaps, remove speckle noise.
    kernel_size = SEGMENTATION.morph_kernel_size | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise SegmentationError(
            "No lesion contour found. Try selecting a tighter crop around "
            "the lesion with more surrounding skin for contrast."
        )

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < SEGMENTATION.min_contour_area:
        raise SegmentationError(
            "Detected region is too small to be a reliable lesion contour. "
            "Try a higher-resolution crop."
        )

    # Rebuild a clean mask containing only the selected contour so that
    # downstream feature extraction isn't polluted by smaller artifacts.
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return SegmentationResult(mask=clean_mask, contour=contour, image_rgb=image_rgb)
