"""
deepguard/detectors/image_detector.py
───────────────────────────────────────
Image deepfake detection pipeline.

Flow:
  1. Validate file (extension, size)
  2. Query each HuggingFace image model in the registry
  3. Optionally query Sightengine (if credentials are set)
  4. Aggregate scores via ensemble → single verdict
  5. Return DetectionResult
"""

from __future__ import annotations

import logging
from pathlib import Path

from deepguard import config
from deepguard.apis import huggingface as hf_api
from deepguard.apis import sightengine as se_api
from deepguard.detectors.metadata_detector import check_c2pa
from deepguard.detectors.ela_detector import run_ela
from deepguard.ensemble import aggregate
from deepguard.utils.file_utils import MediaType, validate_file
from deepguard.utils.report import DetectionResult, ModelScore

logger = logging.getLogger(__name__)


def detect(image_path: Path | str) -> DetectionResult:
    """
    Run deepfake detection on a single image.

    Parameters
    ----------
    image_path : Path or str
        Path to the image file to analyse.

    Returns
    -------
    DetectionResult
        Verdict, fake probability, per-model scores, and metadata.
    """
    path = Path(image_path)
    validate_file(path, MediaType.IMAGE)

    logger.info("Analysing image: %s", path.name)

    # ── Layer 1: Cryptographic Metadata (C2PA) ───────────────────────────
    meta_result = check_c2pa(path)
    metadata_prob = meta_result.confidence if meta_result else None
    if meta_result:
        logger.info("  [C2PA] Found AI metadata: %s", meta_result.evidence)

    # ── Layer 2: Algorithmic Forensics (ELA) ─────────────────────────────
    ela_result = run_ela(path)
    ela_prob = ela_result.anomaly_score
    logger.info("  [ELA] Anomaly score: %.1f%%", ela_prob * 100)

    model_scores: list[ModelScore] = []

    # ── HuggingFace models ───────────────────────────────────────────────
    for model_id, weight in config.IMAGE_MODELS:
        try:
            raw = hf_api.classify_image(model_id, path)
            score = _extract_score(raw, model_id, weight)
            model_scores.append(score)
            logger.info(
                "  [HF] %s → %s (%.1f%%)", model_id, score.label, score.confidence * 100
            )
        except hf_api.HFAPIError as exc:
            logger.warning("  [HF] Skipping %s — %s", model_id, exc)

    # ── Sightengine (optional) ───────────────────────────────────────────
    se_score: float | None = None
    if config.SIGHTENGINE_ENABLED:
        se_response = se_api.check_image(path)
        se_score = se_api.extract_deepfake_score(se_response)
        if se_score is not None:
            logger.info("  [SE] Sightengine deepfake score: %.1f%%", se_score * 100)

    # ── Ensemble ─────────────────────────────────────────────────────────
    verdict, fake_prob, confidence = aggregate(
        model_scores=model_scores, 
        sightengine_score=se_score,
        metadata_prob=metadata_prob,
        ela_prob=ela_prob
    )

    return DetectionResult(
        file_path=str(path.resolve()),
        media_type="image",
        verdict=verdict,
        fake_probability=fake_prob,
        confidence=confidence,
        model_scores=model_scores,
        sightengine_score=se_score,
        metadata={
            "filename": path.name,
            "file_size_bytes": path.stat().st_size,
            "models_queried": [m for m, _ in config.IMAGE_MODELS],
            "sightengine_used": se_score is not None,
            "c2pa_evidence": meta_result.evidence if meta_result else None,
            "ela_score": round(ela_prob, 3),
        },
    )


def _extract_score(raw: list[dict], model_id: str, weight: float) -> ModelScore:
    """
    Parse the raw HuggingFace classification response into a ModelScore.

    HF returns a list sorted by descending score, e.g.:
      [{"label": "Fake", "score": 0.97}, {"label": "Real", "score": 0.03}]
    """
    best = raw[0]  # highest-probability label
    label = hf_api.normalize_label(best["label"])
    confidence = float(best["score"])

    # Inject weight into raw for ensemble._extract_weight()
    annotated_raw = [{"_weight": weight, **item} for item in raw]

    return ModelScore(
        model_id=model_id,
        label=label,
        confidence=confidence,
        raw=annotated_raw,
    )
