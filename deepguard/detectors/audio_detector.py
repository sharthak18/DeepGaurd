"""
deepguard/detectors/audio_detector.py
───────────────────────────────────────
Audio deepfake / AI-generated speech detection pipeline.

Flow:
  1. Validate file (extension, size)
  2. Convert to 16 kHz mono WAV using FFmpeg (CPU-only, required by most models)
  3. Query each HuggingFace audio classification model
  4. Ensemble scores → single verdict
  5. Clean up temporary WAV file
  6. Return DetectionResult

Supported input formats: .wav .mp3 .flac .ogg .aac .m4a .opus .wma
"""

from __future__ import annotations

import logging
from pathlib import Path

from deepguard.utils.file_utils import MediaType, validate_file
from deepguard.utils.report import DetectionResult

logger = logging.getLogger(__name__)


def detect(audio_path: Path | str) -> DetectionResult:
    """
    Run deepfake detection on an audio file.

    Parameters
    ----------
    audio_path : Path or str
        Path to the audio file to analyse.

    Returns
    -------
    DetectionResult
        Verdict, fake probability, per-model scores, and metadata.
    """
    path = Path(audio_path)
    validate_file(path, MediaType.AUDIO)

    logger.info("Analysing audio: %s", path.name)

    # Audio models are currently unsupported on the HF free inference API
    # and local hardware constraints prevent downloading the multi-GB models.
    logger.warning("Audio deepfake detection is temporarily disabled due to lack of free API support.")

    return DetectionResult(
        file_path=str(path.resolve()),
        media_type="audio",
        verdict="UNCERTAIN",
        fake_probability=0.5,
        confidence=0.0,
        model_scores=[],
        sightengine_score=None,
        metadata={
            "filename": path.name,
            "error": "Audio deepfake APIs (HuggingFace free tier) currently do not support the required models. Awaiting future updates or paid API integration."
        },
    )



