"""
tests/test_video.py
────────────────────
Unit tests for the video detection pipeline.

Mocks FFmpeg frame extraction and the image detector to isolate
video-specific logic: frame aggregation, cleanup, empty-frame handling.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepguard.ensemble import aggregate_frames
from deepguard.utils.report import DetectionResult, ModelScore


# ── aggregate_frames tests ───────────────────────────────────────────────

def _fake_image_result(fake_prob: float) -> DetectionResult:
    label = "fake" if fake_prob >= 0.5 else "real"
    return DetectionResult(
        file_path="/tmp/frame.jpg",
        media_type="image",
        verdict="FAKE" if fake_prob >= 0.60 else ("REAL" if fake_prob <= 0.40 else "UNCERTAIN"),
        fake_probability=fake_prob,
        confidence=abs(fake_prob - 0.5) * 2,
        model_scores=[
            ModelScore("test/model", label, fake_prob if label == "fake" else 1 - fake_prob, [])
        ],
        sightengine_score=None,
        metadata={},
    )


def test_aggregate_frames_all_fake():
    results = [_fake_image_result(0.85), _fake_image_result(0.90), _fake_image_result(0.80)]
    verdict, prob, conf = aggregate_frames(results)
    assert verdict == "FAKE"
    assert prob > 0.60


def test_aggregate_frames_all_real():
    results = [_fake_image_result(0.10), _fake_image_result(0.15), _fake_image_result(0.20)]
    verdict, prob, conf = aggregate_frames(results)
    assert verdict == "REAL"
    assert prob < 0.40


def test_aggregate_frames_mixed():
    # Half fake, half real → uncertain
    results = [_fake_image_result(0.80), _fake_image_result(0.20)]
    verdict, prob, conf = aggregate_frames(results)
    # Mean = 0.5 → uncertain
    assert verdict == "UNCERTAIN"


def test_aggregate_frames_empty():
    verdict, prob, conf = aggregate_frames([])
    assert verdict == "UNCERTAIN"
    assert prob == 0.5
    assert conf == 0.0


# ── video_detector.detect() with mocked FFmpeg + image detector ──────────

@patch("deepguard.detectors.video_detector.extract_frames")
@patch("deepguard.detectors.video_detector.image_detector.detect")
@patch("deepguard.detectors.video_detector.require_ffmpeg")
def test_video_detect_fake(mock_require_ffmpeg, mock_img_detect, mock_extract, tmp_path):
    """detect() with 3 fake frames → FAKE verdict."""
    import shutil

    video = tmp_path / "test.mp4"
    video.write_bytes(b"\x00" * 100)

    # Create fake frame files in a temp dir
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    frames = []
    for i in range(3):
        f = frame_dir / f"frame_{i:05d}.jpg"
        f.write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)
        frames.append(f)

    mock_extract.return_value = (frame_dir, frames)
    mock_img_detect.return_value = _fake_image_result(0.88)

    from deepguard.detectors.video_detector import detect
    result = detect(video)

    assert result.verdict == "FAKE"
    assert result.frames_analysed == 3
    assert len(result.frame_results) == 3
    assert not frame_dir.exists()  # cleaned up


@patch("deepguard.detectors.video_detector.extract_frames")
@patch("deepguard.detectors.video_detector.image_detector.detect")
@patch("deepguard.detectors.video_detector.require_ffmpeg")
def test_video_detect_no_frames(mock_require_ffmpeg, mock_img_detect, mock_extract, tmp_path):
    """When extract_frames returns empty list, return UNCERTAIN."""
    from deepguard.utils.video_utils import FFmpegError

    video = tmp_path / "bad.mp4"
    video.write_bytes(b"\x00" * 100)

    # Simulate FFmpegError
    mock_extract.side_effect = FFmpegError("0 frames extracted")

    from deepguard.detectors.video_detector import detect
    from deepguard.utils.video_utils import FFmpegError as FE

    with pytest.raises(FE):
        detect(video)
