"""
tests/test_audio.py
────────────────────
Unit tests for the audio detection pipeline.

Mocks FFmpeg audio conversion and HuggingFace API calls
to test the detect() orchestrator and label normalisation for audio.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepguard.apis.huggingface import normalize_label


# ── Audio-specific label normalisation ───────────────────────────────────

@pytest.mark.parametrize("label,expected", [
    # ASVspoof-style labels
    ("spoof", "fake"),
    ("Spoof", "fake"),
    ("bonafide", "real"),
    ("bona-fide", "real"),
    # Generic labels
    ("fake", "fake"),
    ("real", "real"),
    ("synthetic", "fake"),
    ("natural", "real"),
    ("Human", "real"),
    ("AI", "fake"),
])
def test_audio_label_normalisation(label, expected):
    assert normalize_label(label) == expected


# ── audio_detector.detect() with mocked FFmpeg + HF API ──────────────────

@patch("deepguard.detectors.audio_detector.convert_audio_to_wav")
@patch("deepguard.detectors.audio_detector.hf_api.classify_audio")
def test_audio_detect_fake(mock_classify, mock_convert, tmp_path):
    """detect() for a spoofed/fake audio file."""
    audio = tmp_path / "fake_speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 100)  # minimal WAV header

    # Simulate WAV conversion
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    wav_file = wav_dir / "audio_16k.wav"
    wav_file.write_bytes(b"RIFF" + b"\x00" * 100)
    mock_convert.return_value = (wav_dir, wav_file)

    # HF returns spoof labels
    mock_classify.return_value = [
        {"label": "spoof", "score": 0.93},
        {"label": "bonafide", "score": 0.07},
    ]

    from deepguard.detectors.audio_detector import detect
    result = detect(audio)

    assert result.verdict == "FAKE"
    assert result.fake_probability > 0.60
    assert result.media_type == "audio"
    assert not wav_dir.exists()  # temp dir cleaned up


@patch("deepguard.detectors.audio_detector.convert_audio_to_wav")
@patch("deepguard.detectors.audio_detector.hf_api.classify_audio")
def test_audio_detect_real(mock_classify, mock_convert, tmp_path):
    """detect() for a genuine/real audio file."""
    audio = tmp_path / "real_speech.mp3"
    audio.write_bytes(b"\xff\xfb" + b"\x00" * 100)  # minimal MP3 header

    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    wav_file = wav_dir / "audio_16k.wav"
    wav_file.write_bytes(b"RIFF" + b"\x00" * 100)
    mock_convert.return_value = (wav_dir, wav_file)

    mock_classify.return_value = [
        {"label": "bonafide", "score": 0.88},
        {"label": "spoof", "score": 0.12},
    ]

    from deepguard.detectors.audio_detector import detect
    result = detect(audio)

    assert result.verdict == "REAL"
    assert result.fake_probability < 0.40
    assert result.sightengine_score is None  # SE doesn't support audio


@patch("deepguard.detectors.audio_detector.convert_audio_to_wav")
@patch("deepguard.detectors.audio_detector.hf_api.classify_audio")
def test_audio_detect_api_failure_partial(mock_classify, mock_convert, tmp_path):
    """When one model fails, the other should still produce a result."""
    from deepguard.apis.huggingface import HFAPIError

    audio = tmp_path / "speech.flac"
    audio.write_bytes(b"fLaC" + b"\x00" * 100)

    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    wav_file = wav_dir / "audio_16k.wav"
    wav_file.write_bytes(b"RIFF" + b"\x00" * 100)
    mock_convert.return_value = (wav_dir, wav_file)

    # First call fails, second succeeds
    mock_classify.side_effect = [
        HFAPIError("Model timeout"),
        [{"label": "spoof", "score": 0.85}, {"label": "bonafide", "score": 0.15}],
    ]

    from deepguard.detectors.audio_detector import detect
    result = detect(audio)

    # Should still get a result from the second model
    assert result.verdict in ("FAKE", "REAL", "UNCERTAIN")
    assert len(result.model_scores) >= 1


def test_audio_detect_missing_file(tmp_path):
    from deepguard.utils.file_utils import FileValidationError
    from deepguard.detectors.audio_detector import detect

    with pytest.raises(FileValidationError):
        detect(tmp_path / "nonexistent.mp3")
