"""
tests/test_image.py
────────────────────
Unit tests for the image detection pipeline.

These tests mock the external API calls so no real HF_TOKEN is needed.
They verify the logic of: label normalisation, score extraction, ensemble,
and the detect() orchestrator function.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepguard.apis.huggingface import normalize_label
from deepguard.ensemble import aggregate
from deepguard.utils.report import ModelScore


# ── normalize_label tests ────────────────────────────────────────────────

@pytest.mark.parametrize("label,expected", [
    ("Fake", "fake"),
    ("FAKE", "fake"),
    ("deepfake", "fake"),
    ("AI-Generated", "fake"),
    ("spoof", "fake"),
    ("Real", "real"),
    ("REAL", "real"),
    ("bonafide", "real"),
    ("bona-fide", "real"),
    ("Human", "real"),
    ("genuine", "real"),
])
def test_normalize_label(label, expected):
    assert normalize_label(label) == expected


def test_normalize_label_unknown():
    assert normalize_label("xyz_unknown_label_9999") == "unknown"


# ── ensemble.aggregate tests ─────────────────────────────────────────────

def _ms(label: str, confidence: float, weight: float = 1.0) -> ModelScore:
    return ModelScore(
        model_id="test/model",
        label=label,
        confidence=confidence,
        raw=[{"_weight": weight}],
    )


def test_ensemble_single_fake():
    scores = [_ms("fake", 0.95)]
    verdict, prob, conf = aggregate(scores)
    assert verdict == "FAKE"
    assert prob > 0.60
    assert conf > 0.0


def test_ensemble_single_real():
    scores = [_ms("real", 0.90)]
    verdict, prob, conf = aggregate(scores)
    assert verdict == "REAL"
    assert prob < 0.40


def test_ensemble_uncertain():
    scores = [_ms("fake", 0.52), _ms("real", 0.52)]
    verdict, prob, _ = aggregate(scores)
    assert verdict == "UNCERTAIN"


def test_ensemble_empty():
    verdict, prob, conf = aggregate([])
    assert verdict == "UNCERTAIN"
    assert prob == 0.5
    assert conf == 0.0


def test_ensemble_all_unknown():
    scores = [_ms("unknown", 0.9), _ms("unknown", 0.8)]
    verdict, _, _ = aggregate(scores)
    assert verdict == "UNCERTAIN"


def test_ensemble_with_sightengine_pushes_to_fake():
    # HF says borderline (0.55 fake), SE says clearly fake (0.9)
    scores = [_ms("fake", 0.55)]
    verdict, prob, _ = aggregate(scores, sightengine_score=0.9, se_weight=0.3)
    # Should push verdict to FAKE
    assert verdict == "FAKE"
    assert prob > 0.60


def test_ensemble_weighted():
    # Model A (weight 0.8) says fake 0.9, Model B (weight 0.2) says real 0.9
    scores = [_ms("fake", 0.9, weight=0.8), _ms("real", 0.9, weight=0.2)]
    verdict, prob, _ = aggregate(scores)
    # Weighted fake_prob ≈ (0.9*0.8 + 0.1*0.2)/(0.8+0.2) = 0.74
    assert verdict == "FAKE"
    assert 0.65 < prob < 0.80


# ── image_detector.detect() with mocked APIs ────────────────────────────

@patch("deepguard.detectors.image_detector.hf_api.classify_image")
@patch("deepguard.detectors.image_detector.se_api.check_image")
@patch("deepguard.detectors.image_detector.se_api.extract_deepfake_score")
@patch("deepguard.detectors.image_detector.config.SIGHTENGINE_ENABLED", False)
def test_image_detect_fake(mock_se_score, mock_se_check, mock_hf_classify, tmp_path):
    """Full detect() pipeline with mocked external calls."""
    # Create a dummy image file
    img = tmp_path / "test.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # minimal JPEG header

    # HF returns FAKE with high confidence for all models
    mock_hf_classify.return_value = [
        {"label": "Fake", "score": 0.95},
        {"label": "Real", "score": 0.05},
    ]

    from deepguard.detectors.image_detector import detect
    result = detect(img)

    assert result.verdict == "FAKE"
    assert result.fake_probability > 0.60
    assert len(result.model_scores) > 0
    assert result.sightengine_score is None  # SE disabled


@patch("deepguard.detectors.image_detector.hf_api.classify_image")
@patch("deepguard.detectors.image_detector.config.SIGHTENGINE_ENABLED", False)
def test_image_detect_real(mock_hf_classify, tmp_path):
    img = tmp_path / "real.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    mock_hf_classify.return_value = [
        {"label": "Real", "score": 0.91},
        {"label": "Fake", "score": 0.09},
    ]

    from deepguard.detectors.image_detector import detect
    result = detect(img)

    assert result.verdict == "REAL"
    assert result.fake_probability < 0.40


def test_image_detect_invalid_file(tmp_path):
    from deepguard.utils.file_utils import FileValidationError
    from deepguard.detectors.image_detector import detect

    with pytest.raises(FileValidationError):
        detect(tmp_path / "nonexistent.jpg")
