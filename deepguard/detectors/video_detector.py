"""
deepguard/detectors/video_detector.py
───────────────────────────────────────
Video deepfake detection pipeline.

Flow:
  1. Validate file
  2. Extract up to N frames using FFmpeg (CPU-only, lightweight)
  3. Run each frame through the image detector
  4. Aggregate frame-level results into a single video verdict
  5. Clean up temporary frame files
  6. Return VideoDetectionResult
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from tqdm import tqdm

from deepguard import config
from deepguard.detectors import image_detector
from deepguard.ensemble import aggregate_frames
from deepguard.utils.file_utils import MediaType, validate_file
from deepguard.utils.report import FrameResult, VideoDetectionResult
from deepguard.utils.video_utils import (
    extract_frames,
    get_video_duration,
    require_ffmpeg,
)

logger = logging.getLogger(__name__)


def detect(video_path: Path | str) -> VideoDetectionResult:
    """
    Run deepfake detection on a video file.

    Parameters
    ----------
    video_path : Path or str
        Path to the video file to analyse.

    Returns
    -------
    VideoDetectionResult
        Per-frame breakdown and aggregated verdict.
    """
    path = Path(video_path)
    validate_file(path, MediaType.VIDEO)
    require_ffmpeg()

    logger.info("Analysing video: %s", path.name)

    duration = get_video_duration(path)
    logger.info("  Duration: %s", f"{duration:.1f}s" if duration else "unknown")

    # ── Frame extraction ─────────────────────────────────────────────────
    tmp_dir, frame_paths = extract_frames(
        path,
        fps=1.0,
        max_frames=config.VIDEO_MAX_FRAMES,
    )

    frame_results: list[FrameResult] = []

    try:
        # ── Per-frame analysis ───────────────────────────────────────────
        for idx, frame_path in enumerate(
            tqdm(frame_paths, desc="Analysing frames", unit="frame"), start=1
        ):
            try:
                result = image_detector.detect(frame_path)
                frame_results.append(
                    FrameResult(
                        frame_index=idx,
                        frame_path=str(frame_path),
                        result=result,
                    )
                )
                logger.debug(
                    "  Frame %d/%d: %s (%.1f%%)",
                    idx, len(frame_paths), result.verdict, result.fake_probability * 100,
                )
            except Exception as exc:
                logger.warning("  Frame %d analysis failed (skipped): %s", idx, exc)

    finally:
        # Always clean up temp frames
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.debug("Cleaned up temp dir: %s", tmp_dir)

    if not frame_results:
        return VideoDetectionResult(
            file_path=str(path.resolve()),
            media_type="video",
            verdict="UNCERTAIN",
            fake_probability=0.5,
            confidence=0.0,
            frames_analysed=0,
            frame_results=[],
            metadata={"error": "No frames could be analysed."},
        )

    # ── Aggregate frame verdicts ─────────────────────────────────────────
    frame_image_results = [fr.result for fr in frame_results]
    verdict, fake_prob, confidence = aggregate_frames(frame_image_results)

    return VideoDetectionResult(
        file_path=str(path.resolve()),
        media_type="video",
        verdict=verdict,
        fake_probability=fake_prob,
        confidence=confidence,
        frames_analysed=len(frame_results),
        frame_results=frame_results,
        metadata={
            "filename": path.name,
            "file_size_bytes": path.stat().st_size,
            "duration_seconds": duration,
            "frames_extracted": len(frame_paths),
            "frames_analysed": len(frame_results),
            "max_frames_setting": config.VIDEO_MAX_FRAMES,
        },
    )
