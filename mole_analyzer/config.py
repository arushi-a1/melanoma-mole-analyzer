"""
Central configuration for segmentation and scoring thresholds.

Keeping these in one place makes the analyzer easy to tune/calibrate
against a labeled dataset without hunting through business logic.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentationConfig:
    # Morphological cleanup kernel size (odd integer, pixels)
    morph_kernel_size: int = 5
    # Minimum contour area (in pixels) to be considered a valid lesion
    min_contour_area: int = 150
    # Gaussian blur kernel applied before thresholding to reduce noise
    blur_kernel_size: int = 5


@dataclass(frozen=True)
class ScoringConfig:
    # Asymmetry: 0 = perfectly symmetric, 1 = fully asymmetric
    asymmetry_low: float = 0.15
    asymmetry_high: float = 0.30

    # Border irregularity: normalized deviation from a perfect circle
    # (0 = perfect circle, higher = more irregular)
    border_low: float = 0.20
    border_high: float = 0.40

    # Color variation: number of statistically distinct color clusters
    # found inside the lesion mask
    color_clusters_low: int = 2
    color_clusters_high: int = 4

    # Diameter, in millimeters, above which lesions are flagged
    # (6mm is the traditional ABCDE "pencil eraser" rule of thumb)
    diameter_mm_threshold: float = 6.0


@dataclass(frozen=True)
class CalibrationConfig:
    # Default assumption if the user does not provide a reference object
    # for pixel-to-millimeter calibration. This is intentionally
    # conservative/approximate and clearly flagged as an estimate.
    default_pixels_per_mm: float = 10.0


SEGMENTATION = SegmentationConfig()
SCORING = ScoringConfig()
CALIBRATION = CalibrationConfig()
