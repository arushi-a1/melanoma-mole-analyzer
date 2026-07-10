# Melanoma Mole Analyzer

A Python desktop application that performs basic image analysis of skin lesions using computer vision techniques.

> **Disclaimer**
> This software is intended for educational and research purposes only. It is not a medical device and should not be used for diagnosis.

---

## Overview

This project analyzes an uploaded image of a skin lesion by comparing it against nearby healthy skin. It evaluates several characteristics commonly associated with melanoma using classical image processing techniques.

The application provides:

- Interactive image upload
- Manual lesion selection
- Healthy skin comparison region
- Color difference analysis
- Shape roundness analysis
- Symmetry analysis
- Simple graphical interface built with Tkinter

---

## Features

### Color Analysis

Computes the average RGB color of the lesion and compares it with healthy surrounding skin.

### Roundness Analysis

Calculates lesion circularity using

\[
Circularity=\frac{4\pi A}{P^2}
\]

where

- A = contour area
- P = contour perimeter

### Symmetry Analysis

Uses OpenCV contour matching to estimate lesion symmetry.

---

## Technologies

- Python
- OpenCV
- NumPy
- Pillow
- Tkinter

---

## Installation

```bash
git clone https://github.com/yourusername/melanoma-mole-analyzer.git

cd melanoma-mole-analyzer

pip install -r requirements.txt

python src/mole_analyzer.py
```

---

## How it Works

1. Upload a skin image.
2. Select the mole region.
3. Select a healthy skin region.
4. The application performs image processing and reports:

- Coloration score
- Roundness score
- Symmetry score

---

## Future Improvements

- HSV color segmentation
- Automatic lesion detection
- Deep learning classification
- ABCDE melanoma scoring
- Probability estimation instead of threshold-based decisions

---

## License

MIT License
