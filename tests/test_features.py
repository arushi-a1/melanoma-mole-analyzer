"""
Unit tests for feature extraction, using synthetic images with known
ground-truth shape properties (a circle should score low on asymmetry
and border irregularity; a jagged star should score high).
"""

import numpy as np
import cv2
import pytest

from mole_analyzer.features import (
    compute_asymmetry,
    compute_border_irregularity,
    compute_diameter,
)
from mole_analyzer.segmentation import segment_lesion, SegmentationError


def make_circle_image(size=200, radius=60, color=(60, 40, 30), bg=(210, 180, 160)):
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), radius, color, -1)
    return img


def make_star_image(size=200, color=(60, 40, 30), bg=(210, 180, 160)):
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    center = np.array([size // 2, size // 2])
    outer_r, inner_r = 80, 25
    points = []
    for i in range(10):
        r = outer_r if i % 2 == 0 else inner_r
        angle = np.pi * i / 5
        points.append(center + r * np.array([np.cos(angle), np.sin(angle)]))
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(img, [pts], color)
    return img


def make_offcenter_blob_image(size=200, color=(60, 40, 30), bg=(210, 180, 160)):
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    cv2.ellipse(img, (size // 2 + 30, size // 2), (50, 20), 20, 0, 360, color, -1)
    cv2.circle(img, (size // 2 - 20, size // 2 - 20), 15, color, -1)
    return img


class TestSegmentation:
    def test_segments_a_clear_circle(self):
        img = make_circle_image()
        result = segment_lesion(img)
        assert result.mask.sum() > 0
        area = cv2.contourArea(result.contour)
        expected_area = np.pi * 60 ** 2
        assert abs(area - expected_area) / expected_area < 0.25

    def test_raises_on_blank_image(self):
        blank = np.full((100, 100, 3), 200, dtype=np.uint8)
        with pytest.raises(SegmentationError):
            segment_lesion(blank)


class TestAsymmetry:
    def test_circle_is_near_symmetric(self):
        img = make_circle_image()
        seg = segment_lesion(img)
        score = compute_asymmetry(seg.mask)
        assert score < 0.1

    def test_irregular_blob_is_more_asymmetric_than_circle(self):
        circle_img = make_circle_image()
        blob_img = make_offcenter_blob_image()

        circle_score = compute_asymmetry(segment_lesion(circle_img).mask)
        blob_score = compute_asymmetry(segment_lesion(blob_img).mask)

        assert blob_score > circle_score


class TestBorderIrregularity:
    def test_circle_has_low_irregularity(self):
        img = make_circle_image()
        seg = segment_lesion(img)
        score = compute_border_irregularity(seg.contour)
        assert score < 0.15

    def test_star_has_higher_irregularity_than_circle(self):
        circle_score = compute_border_irregularity(segment_lesion(make_circle_image()).contour)
        star_score = compute_border_irregularity(segment_lesion(make_star_image()).contour)
        assert star_score > circle_score


class TestDiameter:
    def test_diameter_scales_with_calibration(self):
        img = make_circle_image(radius=60)
        seg = segment_lesion(img)
        px, mm_default, estimated = compute_diameter(seg.contour, pixels_per_mm=None)
        _, mm_calibrated, not_estimated = compute_diameter(seg.contour, pixels_per_mm=20.0)

        assert estimated is True
        assert not_estimated is False
        assert px == pytest.approx(120, abs=5)
        assert mm_calibrated == pytest.approx(px / 20.0)
        assert mm_default != mm_calibrated
