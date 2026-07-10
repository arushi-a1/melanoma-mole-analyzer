"""
High-level orchestration: takes a cropped lesion image, runs segmentation
and ABCD(E) feature extraction, and produces a structured, human-readable
result with a risk banding (NOT a diagnosis -- see README/disclaimer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
from PIL import Image

from .config import SCORING
from .features import ABCDFeatures, extract_abcd_features
from .segmentation import SegmentationError, segment_lesion


class RiskBand(str, Enum):
    LOW = "Low concern"
    MODERATE = "Moderate concern"
    ELEVATED = "Elevated concern"


@dataclass
class CriterionVerdict:
    name: str
    score: float
    band: RiskBand
    detail: str


@dataclass
class AnalysisResult:
    timestamp: str
    features: ABCDFeatures
    criteria: list[CriterionVerdict]
    overall_band: RiskBand
    overall_score: float  # 0-1 weighted composite, higher = more concerning
    notes: list[str] = field(default_factory=list)


def _band_from_thresholds(value: float, low: float, high: float) -> RiskBand:
    if value <= low:
        return RiskBand.LOW
    if value <= high:
        return RiskBand.MODERATE
    return RiskBand.ELEVATED


class MoleAnalyzer:
    """
    Main entry point for analyzing a single lesion image.

    Example:
        analyzer = MoleAnalyzer()
        result = analyzer.analyze(pil_image)
        print(result.overall_band, result.overall_score)
    """

    def __init__(self, pixels_per_mm: Optional[float] = None):
        self.pixels_per_mm = pixels_per_mm

    def analyze(self, image: Image.Image) -> AnalysisResult:
        image_rgb = np.array(image.convert("RGB"))

        seg = segment_lesion(image_rgb)
        features = extract_abcd_features(
            image_rgb, seg.mask, seg.contour, self.pixels_per_mm
        )

        criteria = [
            CriterionVerdict(
                name="Asymmetry",
                score=features.asymmetry_score,
                band=_band_from_thresholds(
                    features.asymmetry_score, SCORING.asymmetry_low, SCORING.asymmetry_high
                ),
                detail=f"{features.asymmetry_score:.2f} (0 = symmetric, 1 = fully asymmetric)",
            ),
            CriterionVerdict(
                name="Border irregularity",
                score=features.border_irregularity_score,
                band=_band_from_thresholds(
                    features.border_irregularity_score, SCORING.border_low, SCORING.border_high
                ),
                detail=f"{features.border_irregularity_score:.2f} (0 = perfectly smooth/circular)",
            ),
            CriterionVerdict(
                name="Color variation",
                score=features.color_variation_score,
                band=_band_from_thresholds(
                    features.color_variation_score,
                    (SCORING.color_clusters_low - 1) / 4,
                    (SCORING.color_clusters_high - 1) / 4,
                ),
                detail=(
                    f"{features.color_cluster_count} distinct color cluster(s) detected"
                ),
            ),
            CriterionVerdict(
                name="Diameter",
                score=min(features.diameter_mm / SCORING.diameter_mm_threshold, 2.0) / 2,
                band=(
                    RiskBand.LOW
                    if features.diameter_mm <= SCORING.diameter_mm_threshold
                    else RiskBand.ELEVATED
                ),
                detail=(
                    f"{features.diameter_mm:.1f} mm"
                    + (" (estimated, no calibration provided)" if features.diameter_is_estimated else "")
                ),
            ),
        ]

        weights = {"Asymmetry": 0.3, "Border irregularity": 0.25, "Color variation": 0.25, "Diameter": 0.2}
        overall_score = sum(c.score * weights[c.name] for c in criteria)
        overall_band = _band_from_thresholds(overall_score, 0.25, 0.45)

        notes = []
        if features.diameter_is_estimated:
            notes.append(
                "Diameter is an estimate. For an accurate measurement, include a "
                "reference object of known size (e.g. a ruler) in the photo and "
                "pass its pixels-per-mm scale to MoleAnalyzer."
            )
        notes.append(
            "This tool provides an automated image-processing approximation of "
            "the ABCDE rule for educational purposes only. It is not a diagnosis. "
            "Any lesion of concern should be evaluated by a dermatologist."
        )

        return AnalysisResult(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            features=features,
            criteria=criteria,
            overall_band=overall_band,
            overall_score=overall_score,
            notes=notes,
        )

    def analyze_pair(
        self, earlier: Image.Image, later: Image.Image
    ) -> tuple[AnalysisResult, AnalysisResult, dict]:
        """
        Runs analysis on two images of the same lesion taken at different
        times and reports a simple "Evolution" delta -- the E in ABCDE.
        """
        result_a = self.analyze(earlier)
        result_b = self.analyze(later)

        delta = {
            "asymmetry_delta": result_b.features.asymmetry_score - result_a.features.asymmetry_score,
            "border_delta": result_b.features.border_irregularity_score
            - result_a.features.border_irregularity_score,
            "color_cluster_delta": result_b.features.color_cluster_count
            - result_a.features.color_cluster_count,
            "diameter_mm_delta": result_b.features.diameter_mm - result_a.features.diameter_mm,
            "overall_score_delta": result_b.overall_score - result_a.overall_score,
        }
        return result_a, result_b, delta
