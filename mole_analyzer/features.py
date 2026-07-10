"""
Feature extraction implementing an approximation of the dermatological
ABCDE rule:

    A - Asymmetry
    B - Border irregularity
    C - Color variation
    D - Diameter
    E - Evolution (change over time; requires multiple images, see analyzer)

Each function operates on a segmented lesion (mask + contour) and returns
a normalized score, documented per-function.

Note on the original prototype's symmetry code: it computed
`cv2.matchShapes(contour[:h//2], contour[h//2:], ...)`, which slices the
*array of contour points* in half rather than splitting the lesion shape
spatially. That comparison is close to meaningless. The implementation
below instead flips the actual lesion mask across its own centroid axes
and measures pixel-level overlap, which is what "symmetry" should mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import warnings

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from .config import CALIBRATION


@dataclass
class ABCDFeatures:
    asymmetry_score: float          # 0 (symmetric) - 1 (asymmetric)
    border_irregularity_score: float  # 0 (smooth circle) - 1+ (irregular)
    color_cluster_count: int        # number of distinct color clusters
    color_variation_score: float    # 0-1, based on cluster count + variance
    diameter_px: float
    diameter_mm: float
    diameter_is_estimated: bool     # True if no calibration was provided


def compute_asymmetry(mask: np.ndarray) -> float:
    """
    Measures shape asymmetry by flipping the lesion mask across the
    vertical and horizontal axes through its centroid and computing the
    fraction of non-overlapping area (a Jaccard-distance-like measure),
    averaged across both axes.

    Returns a value in [0, 1]; 0 = perfectly symmetric.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 1.0

    cx, cy = xs.mean(), ys.mean()
    h, w = mask.shape

    # Vertical-axis flip (left-right mirror about centroid x)
    m1 = cv2.getRotationMatrix2D((cx, cy), 0, 1)
    flipped_lr = cv2.flip(mask, 1)
    # Re-align the flipped image so its centroid matches the original
    shift_x = int(round(2 * (cx - w / 2)))
    flipped_lr = np.roll(flipped_lr, shift_x, axis=1)

    flipped_tb = cv2.flip(mask, 0)
    shift_y = int(round(2 * (cy - h / 2)))
    flipped_tb = np.roll(flipped_tb, shift_y, axis=0)

    def overlap_distance(a: np.ndarray, b: np.ndarray) -> float:
        a_bool = a > 0
        b_bool = b > 0
        union = np.logical_or(a_bool, b_bool).sum()
        if union == 0:
            return 1.0
        intersection = np.logical_and(a_bool, b_bool).sum()
        iou = intersection / union
        return 1.0 - iou

    d_vertical = overlap_distance(mask, flipped_lr)
    d_horizontal = overlap_distance(mask, flipped_tb)
    return float(np.clip((d_vertical + d_horizontal) / 2, 0.0, 1.0))


def compute_border_irregularity(contour: np.ndarray) -> float:
    """
    Measures how much the lesion's border deviates from a perfect circle
    using compactness: a perfect circle has compactness == 1; increasing
    irregularity (notches, jagged edges) pushes perimeter^2 up relative
    to area, so compactness drops. We return 1 - compactness, clipped to
    [0, +inf), so 0 = perfect circle and larger values = more irregular.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area <= 0 or perimeter <= 0:
        return 1.0

    compactness = (4 * np.pi * area) / (perimeter ** 2)  # 1.0 for a circle
    irregularity = 1.0 - compactness
    return float(max(0.0, irregularity))


def compute_color_variation(image_rgb: np.ndarray, mask: np.ndarray, max_clusters: int = 5) -> tuple[int, float]:
    """
    Clusters the colors of pixels inside the lesion mask with k-means
    (trying k=1..max_clusters and picking the elbow via inertia drop-off)
    to approximate the dermatological notion of "number of distinct
    colors present" (tan, brown, black, red, white, blue-gray).

    Returns (cluster_count, variation_score) where variation_score is a
    0-1 normalized value combining cluster count and color variance.
    """
    pixels = image_rgb[mask > 0].astype(np.float32)
    n_unique = len(np.unique(pixels.reshape(-1, 3), axis=0))
    effective_max_k = max(1, min(max_clusters, n_unique))
    if len(pixels) < effective_max_k or effective_max_k == 1:
        return effective_max_k, 0.0

    inertias = []
    labels_by_k = {}
    for k in range(1, effective_max_k + 1):
        km = KMeans(n_clusters=k, n_init=4, random_state=42)
        km.fit(pixels)
        inertias.append(km.inertia_)
        labels_by_k[k] = km

    # Simple elbow heuristic: pick the smallest k where adding another
    # cluster reduces inertia by less than 15% of the first-step drop.
    if len(inertias) == 1:
        chosen_k = 1
    else:
        first_drop = max(inertias[0] - inertias[1], 1e-6)
        chosen_k = effective_max_k
        for k in range(2, effective_max_k + 1):
            drop = inertias[k - 2] - (inertias[k - 1] if k - 1 < len(inertias) else inertias[-1])
            if drop < 0.15 * first_drop:
                chosen_k = k - 1
                break

    chosen_k = max(1, chosen_k)
    variance = float(np.mean(np.var(pixels, axis=0)))  # overall color spread
    # Normalize: 1 cluster/low variance -> ~0, many clusters/high variance -> ~1
    cluster_component = min(chosen_k / max_clusters, 1.0)
    variance_component = min(variance / 3000.0, 1.0)  # empirically scaled
    variation_score = float(np.clip(0.6 * cluster_component + 0.4 * variance_component, 0.0, 1.0))

    return chosen_k, variation_score


def compute_diameter(contour: np.ndarray, pixels_per_mm: float | None = None) -> tuple[float, float, bool]:
    """
    Estimates lesion diameter as the diameter of the minimum enclosing
    circle around the contour.

    Args:
        contour: lesion contour in pixel coordinates.
        pixels_per_mm: calibration factor from a reference object placed
            in-frame (e.g. a ruler or coin of known size). If None, a
            conservative default calibration is used and the result is
            flagged as estimated.

    Returns:
        (diameter_px, diameter_mm, is_estimated)
    """
    (_, _), radius = cv2.minEnclosingCircle(contour)
    diameter_px = float(radius * 2)

    is_estimated = pixels_per_mm is None
    scale = pixels_per_mm or CALIBRATION.default_pixels_per_mm
    diameter_mm = diameter_px / scale

    return diameter_px, diameter_mm, is_estimated


def extract_abcd_features(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    contour: np.ndarray,
    pixels_per_mm: float | None = None,
) -> ABCDFeatures:
    """Convenience wrapper that runs all feature extractors at once."""
    asymmetry = compute_asymmetry(mask)
    border = compute_border_irregularity(contour)
    n_clusters, color_score = compute_color_variation(image_rgb, mask)
    diameter_px, diameter_mm, is_estimated = compute_diameter(contour, pixels_per_mm)

    return ABCDFeatures(
        asymmetry_score=asymmetry,
        border_irregularity_score=border,
        color_cluster_count=n_clusters,
        color_variation_score=color_score,
        diameter_px=diameter_px,
        diameter_mm=diameter_mm,
        diameter_is_estimated=is_estimated,
    )
