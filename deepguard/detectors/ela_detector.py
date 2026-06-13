"""
deepguard/detectors/ela_detector.py
─────────────────────────────────────
Layer 2 Forensics: Error Level Analysis (ELA)

Algorithmic (non-AI) approach to detect manual image manipulation (Photoshop,
face-swaps, splicing).

How it works:
1. Resaves the image at a known JPEG compression quality (e.g., 90%).
2. Calculates the absolute difference between the original and the resaved image.
3. In an untouched image, the compression error is generally uniform.
4. If a piece of another image was pasted in (spliced), its error level will
   differ significantly from the background, creating high variance/anomalies.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

logger = logging.getLogger(__name__)

class ELAResult:
    def __init__(self, anomaly_score: float, is_manipulated: bool, evidence: str | None = None):
        """
        anomaly_score: 0.0 to 1.0 (higher means more variance/manipulation)
        is_manipulated: boolean threshold
        """
        self.anomaly_score = anomaly_score
        self.is_manipulated = is_manipulated
        self.evidence = evidence

def run_ela(image_path: Path, quality: int = 90) -> ELAResult:
    """
    Run Error Level Analysis on the image.
    Returns an ELAResult with an anomaly score.
    """
    try:
        original = Image.open(image_path).convert('RGB')

        # We need a temporary file to save the compressed version
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        original.save(tmp_path, 'JPEG', quality=quality)
        compressed = Image.open(tmp_path).convert('RGB')

        # Calculate absolute difference
        diff = ImageChops.difference(original, compressed)

        # Enhance the difference so it's easier to measure mathematically
        # Multiply by a factor (extrema is max difference found)
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])

        if max_diff == 0:
            # Completely identical (e.g., it was already a perfect q=90 jpeg, or PNG)
            # Low anomaly.
            Path(tmp_path).unlink(missing_ok=True)
            return ELAResult(0.0, False, "Uniform compression (no anomalies detected).")

        # We look at the variance of the difference.
        # High variance means some parts of the image have wildly different
        # compression histories than others (classic sign of splicing).
        stat = ImageStat.Stat(diff)

        # stat.stddev gives the standard deviation for R, G, B channels
        # Average the standard deviation across channels
        avg_stddev = sum(stat.stddev) / len(stat.stddev)

        # Normalize the score (this threshold is heuristic based on standard ELA)
        # Usually, untouched images have stddev < 5. Heavily manipulated > 15.
        score = min(avg_stddev / 20.0, 1.0)

        is_manipulated = score > 0.6  # If stddev > 12, highly likely spliced

        evidence = None
        if is_manipulated:
            evidence = f"High compression variance detected (ELA Score: {score:.2f}). Indicates manual splicing/photoshop."
            logger.info("ELA Anomaly Detected: %s", evidence)

        Path(tmp_path).unlink(missing_ok=True)
        return ELAResult(score, is_manipulated, evidence)

    except Exception as e:
        logger.warning("ELA failed for %s: %s", image_path.name, e)
        return ELAResult(0.0, False, f"ELA check failed: {e}")
