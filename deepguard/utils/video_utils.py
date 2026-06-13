"""
deepguard/utils/video_utils.py
────────────────────────────────
Lightweight, CPU-only video frame extraction using FFmpeg.

FFmpeg is free, open-source, and runs efficiently even on i3 hardware.
It does NOT require GPU. Frames are extracted to a temporary directory
and cleaned up after use.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class FFmpegError(RuntimeError):
    """Raised when FFmpeg is not installed or frame extraction fails."""


def require_ffmpeg() -> None:
    """Raise FFmpegError if ffmpeg is not on PATH."""
    if not shutil.which("ffmpeg"):
        raise FFmpegError(
            "ffmpeg is not installed or not on PATH.\n"
            "  Install: sudo apt install ffmpeg  (Ubuntu/Debian)\n"
            "           sudo dnf install ffmpeg  (Fedora)\n"
            "           brew install ffmpeg      (macOS)"
        )


def extract_frames(
    video_path: Path,
    *,
    fps: float = 1.0,
    max_frames: int = 10,
    quality: int = 3,
) -> tuple[Path, list[Path]]:
    """
    Extract frames from a video file using FFmpeg.

    Parameters
    ----------
    video_path : Path
        Path to the input video file.
    fps : float
        How many frames to extract per second of video. Default: 1.0
    max_frames : int
        Maximum number of frames to keep. Default: 10
        (caps API calls to avoid burning through rate limits)
    quality : int
        JPEG quality scale 2–5 (lower = higher quality). Default: 3

    Returns
    -------
    temp_dir : Path
        The temporary directory holding the frames (caller must delete it).
    frame_paths : list[Path]
        Sorted list of extracted frame image paths.

    Raises
    ------
    FFmpegError
        If ffmpeg is not installed or extraction fails.
    """
    require_ffmpeg()

    tmp = Path(tempfile.mkdtemp(prefix="deepguard_frames_"))
    logger.debug("Extracting frames from %s → %s (fps=%.1f)", video_path, tmp, fps)

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", str(quality),
        "-frames:v", str(max_frames),   # hard cap
        str(tmp / "frame_%05d.jpg"),
        "-loglevel", "error",
        "-y",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise FFmpegError(
            f"FFmpeg frame extraction failed for {video_path.name!r}:\n"
            f"{result.stderr.strip()}"
        )

    frames = sorted(tmp.glob("frame_*.jpg"))
    logger.info("Extracted %d frames from %s", len(frames), video_path.name)

    if not frames:
        shutil.rmtree(tmp, ignore_errors=True)
        raise FFmpegError(
            f"FFmpeg extracted 0 frames from {video_path.name!r}. "
            "The file may be corrupt or not contain a video stream."
        )

    return tmp, frames


def get_video_duration(video_path: Path) -> float | None:
    """
    Get video duration in seconds using ffprobe.
    Returns None if ffprobe is unavailable or fails.
    """
    if not shutil.which("ffprobe"):
        return None

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def convert_audio_to_wav(audio_path: Path) -> tuple[Path, Path]:
    """
    Convert an audio file to 16kHz mono WAV format suitable for speech models.

    Returns (temp_dir, wav_path). Caller is responsible for deleting temp_dir.
    If the file is already a .wav, it is still normalised (sample rate, channels).
    """
    require_ffmpeg()

    tmp = Path(tempfile.mkdtemp(prefix="deepguard_audio_"))
    out_path = tmp / "audio_16k.wav"

    cmd = [
        "ffmpeg",
        "-i", str(audio_path),
        "-ar", "16000",   # 16 kHz sample rate
        "-ac", "1",       # mono
        "-c:a", "pcm_s16le",
        str(out_path),
        "-loglevel", "error",
        "-y",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise FFmpegError(
            f"FFmpeg audio conversion failed for {audio_path.name!r}:\n"
            f"{result.stderr.strip()}"
        )

    logger.debug("Converted %s → %s", audio_path.name, out_path)
    return tmp, out_path
