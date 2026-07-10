"""
Entry point for the Mole Analyzer project.

Usage:
    python main.py                     # launch the GUI
    python main.py --cli path.jpg      # run a headless analysis on an image
    python main.py --cli path.jpg --ppm 12.5 --out results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from mole_analyzer.analyzer import MoleAnalyzer
from mole_analyzer.report import save_annotated_overlay, save_json_report, save_text_summary
from mole_analyzer.segmentation import SegmentationError


def run_cli(image_path: str, pixels_per_mm: float | None, out_dir: str):
    image = Image.open(image_path)
    analyzer = MoleAnalyzer(pixels_per_mm=pixels_per_mm)

    try:
        result = analyzer.analyze(image)
    except SegmentationError as e:
        print(f"Segmentation failed: {e}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    save_json_report(result, out_path / "mole_report.json")
    save_text_summary(result, out_path / "mole_report.txt")
    save_annotated_overlay(image, result, out_path / "mole_overlay.png")

    print(f"Overall: {result.overall_band.value} ({result.overall_score:.2f}/1.00)")
    for c in result.criteria:
        print(f"  - {c.name}: {c.band.value} -- {c.detail}")
    print(f"\nReport written to {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Mole Analyzer")
    parser.add_argument("--cli", metavar="IMAGE_PATH", help="Run headless analysis on a single lesion image")
    parser.add_argument("--ppm", type=float, default=None, help="Calibration: pixels per millimeter")
    parser.add_argument("--out", default="output", help="Output directory for CLI report (default: ./output)")
    args = parser.parse_args()

    if args.cli:
        run_cli(args.cli, args.ppm, args.out)
    else:
        from mole_analyzer.gui import run
        run()


if __name__ == "__main__":
    main()
