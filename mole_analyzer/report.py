"""
Report generation: turn an AnalysisResult into artifacts a user can keep --
a JSON record, a plain-text summary, and an annotated image showing the
detected lesion contour.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .analyzer import AnalysisResult
from .segmentation import segment_lesion


def result_to_dict(result: AnalysisResult) -> dict:
    d = asdict(result)
    # Enum -> string for JSON serialization
    d["overall_band"] = result.overall_band.value
    for c in d["criteria"]:
        c["band"] = c["band"].value if hasattr(c["band"], "value") else c["band"]
    return d


def save_json_report(result: AnalysisResult, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(result_to_dict(result), indent=2))
    return path


def save_text_summary(result: AnalysisResult, path: str | Path) -> Path:
    path = Path(path)
    lines = [
        "Mole Analyzer Report",
        "=====================",
        f"Generated: {result.timestamp}",
        f"Overall assessment: {result.overall_band.value} (score {result.overall_score:.2f}/1.00)",
        "",
        "Criteria:",
    ]
    for c in result.criteria:
        lines.append(f"  - {c.name}: {c.band.value} -- {c.detail}")
    lines.append("")
    lines.append("Notes:")
    for n in result.notes:
        lines.append(f"  * {n}")
    path.write_text("\n".join(lines))
    return path


def save_annotated_overlay(image: Image.Image, result: AnalysisResult, path: str | Path) -> Path:
    """
    Draws the detected lesion contour and a minimum enclosing circle
    (used for the diameter estimate) on top of the original image, so
    users can visually sanity-check what the analyzer actually measured.
    """
    image_rgb = np.array(image.convert("RGB"))
    seg = segment_lesion(image_rgb)

    overlay = image_rgb.copy()
    cv2.drawContours(overlay, [seg.contour], -1, (0, 255, 0), 2)

    (cx, cy), radius = cv2.minEnclosingCircle(seg.contour)
    cv2.circle(overlay, (int(cx), int(cy)), int(radius), (255, 0, 0), 1)

    band_color = {
        "Low concern": (0, 200, 0),
        "Moderate concern": (230, 165, 0),
        "Elevated concern": (220, 0, 0),
    }.get(result.overall_band.value, (0, 0, 0))

    cv2.putText(
        overlay,
        result.overall_band.value,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        band_color,
        2,
        cv2.LINE_AA,
    )

    out_path = Path(path)
    Image.fromarray(overlay).save(out_path)
    return out_path
