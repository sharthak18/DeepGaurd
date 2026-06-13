"""
deepguard/ensemble.py
──────────────────────
Score aggregation across multiple detection models.

Combines results from:
  - Multiple HuggingFace image/audio models (weighted average)
  - Optional Sightengine score

Produces a single verdict: FAKE | REAL | UNCERTAIN
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepguard.utils.report import ModelScore

logger = logging.getLogger(__name__)

# Verdict thresholds
_FAKE_THRESHOLD = 0.60     # fake_prob >= 0.60 → FAKE
_REAL_THRESHOLD = 0.40     # fake_prob <= 0.40 → REAL
# 0.40 < fake_prob < 0.60 → UNCERTAIN


def aggregate(
    model_scores: list["ModelScore"],
    sightengine_score: float | None = None,
    metadata_prob: float | None = None,
    ela_prob: float | None = None,
    se_weight: float = 0.70,  # Open-source models have high false-positives on real videos; Sightengine is required to balance them.
) -> tuple[str, float, float]:
    """
    Aggregate scores from multiple models into a single verdict.

    Parameters
    ----------
    model_scores : list[ModelScore]
        Each score object has .label ('fake'|'real'|'unknown') and .confidence [0–1].
    sightengine_score : float | None
        Sightengine fake probability [0–1], or None if not available.
    se_weight : float
        Weight given to Sightengine in the ensemble (default: 0.20).

    Returns
    -------
    verdict : str
        'FAKE' | 'REAL' | 'UNCERTAIN'
    fake_probability : float
        Ensemble fake probability [0.0 – 1.0].
    confidence : float
        How confident we are in the verdict [0.0 – 1.0].
    """
    if metadata_prob == 1.0:
        # Cryptographic proof of AI generation. Instant override.
        logger.info("Ensemble short-circuit: C2PA metadata proves image is FAKE.")
        return "FAKE", 1.0, 1.0

    if not model_scores and sightengine_score is None and ela_prob is None:
        logger.warning("ensemble.aggregate called with no scores available.")
        return "UNCERTAIN", 0.5, 0.0

    # Convert each model score to a fake probability
    weighted_sum = 0.0
    total_weight = 0.0

    for ms in model_scores:
        if ms.label == "unknown":
            continue  # Skip unusable scores

        # Extract weight from model metadata (stored as raw[0]["weight"] if present)
        # Otherwise fall back to equal weighting
        weight = _extract_weight(ms)

        if ms.label == "fake":
            prob = ms.confidence
        else:  # 'real'
            prob = 1.0 - ms.confidence

        weighted_sum += prob * weight
        total_weight += weight

    if total_weight == 0:
        logger.warning("All model scores had unknown labels — returning UNCERTAIN.")
        return "UNCERTAIN", 0.5, 0.0

    hf_fake_prob = weighted_sum / total_weight if total_weight > 0 else 0.5

    fake_prob = hf_fake_prob

    # 1. Incorporate Sightengine if available (e.g. 20% weight)
    if sightengine_score is not None:
        fake_prob = fake_prob * (1.0 - se_weight) + sightengine_score * se_weight

    # 2. Incorporate ELA if available and significant
    # If ELA score is high (anomaly), it strongly pulls the probability towards FAKE.
    if ela_prob is not None:
        if ela_prob > 0.6:
            # High anomaly: strong indicator of splicing
            fake_prob = max(fake_prob, ela_prob)
        elif ela_prob < 0.2:
            # Very low anomaly: pulls slightly towards real
            fake_prob = fake_prob * 0.9

    # Determine verdict
    if fake_prob >= _FAKE_THRESHOLD:
        verdict = "FAKE"
    elif fake_prob <= _REAL_THRESHOLD:
        verdict = "REAL"
    else:
        verdict = "UNCERTAIN"

    # Confidence: distance from the 0.5 midpoint, scaled to [0, 1]
    confidence = min(abs(fake_prob - 0.5) * 2.0, 1.0)

    logger.debug(
        "Ensemble result: verdict=%s fake_prob=%.3f confidence=%.3f",
        verdict, fake_prob, confidence,
    )
    return verdict, round(fake_prob, 4), round(confidence, 4)


def aggregate_frames(
    frame_results: list,  # list[DetectionResult]
) -> tuple[str, float, float]:
    """
    Aggregate per-frame image results into a single video verdict.

    Strategy:
    - Weighted average of fake_probability across all frames.
    - Frames are weighted equally (simple mean).
    """
    if not frame_results:
        return "UNCERTAIN", 0.5, 0.0

    probs = [fr.fake_probability for fr in frame_results]
    fake_prob = sum(probs) / len(probs)

    if fake_prob >= _FAKE_THRESHOLD:
        verdict = "FAKE"
    elif fake_prob <= _REAL_THRESHOLD:
        verdict = "REAL"
    else:
        verdict = "UNCERTAIN"

    confidence = min(abs(fake_prob - 0.5) * 2.0, 1.0)
    return verdict, round(fake_prob, 4), round(confidence, 4)


def _extract_weight(ms: "ModelScore") -> float:
    """
    Extract the model weight from the raw response metadata, if stored.
    Falls back to 1.0 (equal weighting).
    """
    if ms.raw and isinstance(ms.raw, list) and len(ms.raw) > 0:
        first = ms.raw[0]
        if isinstance(first, dict) and "_weight" in first:
            return float(first["_weight"])
    return 1.0
