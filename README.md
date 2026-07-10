# Mole Analyzer

[![Tests](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/tests.yml/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/tests.yml)

An image-processing toolkit that approximates the dermatological **ABCDE rule**
(Asymmetry, Border irregularity, Color variation, Diameter, Evolution) for
skin lesion photos, with a desktop GUI, a CLI, an automated test suite, and
exportable reports.

> ⚠️ **Disclaimer:** This project is an educational computer-vision
> portfolio piece. It is **not a medical device**, has **not been
> clinically validated**, and does **not diagnose melanoma or any other
> condition**. It exists to demonstrate image segmentation, feature
> engineering, and software architecture skills. If you have a lesion
> that concerns you, see a dermatologist.

## Why this exists

A naive version of this idea checks whether pixels are darker than a fixed
RGB threshold and calls that a "mole." That approach breaks on real photos
(lighting, skin tone, hair, shadows) and doesn't measure anything a
dermatologist would recognize. This version instead:

- **Segments** the lesion from skin using Otsu thresholding in L\*a\*b\*
  color space plus morphological cleanup, rather than a flat RGB cutoff.
- Implements each ABCDE criterion as its own, independently testable
  algorithm:
  - **Asymmetry** — mirrors the segmented lesion mask across its own
    centroid axes and measures IoU overlap loss.
  - **Border irregularity** — circularity/compactness of the contour
    (`4π·area / perimeter²`), not a coordinate-array slicing trick.
  - **Color variation** — k-means clustering of in-lesion pixel colors
    with an elbow heuristic to estimate the number of distinct colors.
  - **Diameter** — minimum enclosing circle, converted to millimeters via
    optional pixel-to-mm calibration.
  - **Evolution** — `analyze_pair()` diffs two analyses of the same
    lesion taken at different times.
- Produces a **weighted risk band** (Low / Moderate / Elevated concern)
  rather than a binary "cancer or not" call, and clearly labels every
  score with what it does and doesn't mean.
- Ships with **unit tests** against synthetic shapes with known
  ground-truth geometry (circles vs. stars vs. off-center blobs).

## Project structure

```
mole_analyzer_project/
├── main.py                  # CLI + GUI entry point
├── mole_analyzer/
│   ├── __init__.py
│   ├── config.py             # Tunable thresholds, all in one place
│   ├── segmentation.py       # Lesion segmentation (Otsu on L*a*b*)
│   ├── features.py           # ABCD feature extraction algorithms
│   ├── analyzer.py           # Orchestration, scoring, risk banding
│   ├── report.py             # JSON / text / annotated-image export
│   └── gui.py                # Tkinter desktop UI
├── tests/
│   └── test_features.py      # Pytest suite with synthetic test images
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### GUI

```bash
python main.py
```

Upload an image, click once to start a selection box around the lesion,
click again to finish it, then hit **Analyze Selection**. Results can be
exported as a JSON record, a text summary, and an annotated overlay image
via **Export Report**.

### CLI

```bash
python main.py --cli path/to/lesion.jpg --out results/
```

Add `--ppm <pixels_per_mm>` if you have a calibration reference (e.g. a
ruler photographed at the same distance) for an accurate diameter in mm;
otherwise the diameter is reported as an estimate.

### As a library

```python
from PIL import Image
from mole_analyzer import MoleAnalyzer

analyzer = MoleAnalyzer(pixels_per_mm=12.5)  # omit if uncalibrated
result = analyzer.analyze(Image.open("lesion.jpg"))

print(result.overall_band, result.overall_score)
for criterion in result.criteria:
    print(criterion.name, criterion.band, criterion.detail)
```

## Testing

```bash
pytest tests/ -v
```

Tests validate feature extraction against synthetic images with known
geometry (e.g. a perfect circle must score near-zero asymmetry and
border irregularity; a jagged star must score higher on both).

## Possible extensions

- Swap the k-means color clustering for a proper Gaussian Mixture Model
  with BIC-based cluster selection.
- Replace Otsu segmentation with a trained U-Net for lesions with low
  contrast against skin.
- Add a `history.py` module that stores multiple `analyze_pair()` runs
  per lesion over time for real evolution tracking.
- Calibrate risk-band thresholds against a labeled dataset (e.g. ISIC)
  and report precision/recall instead of hand-picked cutoffs.

## License

MIT — see `LICENSE`.
