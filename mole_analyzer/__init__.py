"""
mole_analyzer
=============

A lesion-image analysis toolkit that implements a simplified, image-processing
based approximation of the dermatological ABCDE rule (Asymmetry, Border
irregularity, Color variation, Diameter, and a placeholder for Evolution).

IMPORTANT: This project is an educational computer-vision portfolio piece.
It is NOT a medical device and does NOT provide medical diagnoses. See
README.md for the full disclaimer.
"""

from .analyzer import MoleAnalyzer, AnalysisResult

__all__ = ["MoleAnalyzer", "AnalysisResult"]
__version__ = "2.0.0"
