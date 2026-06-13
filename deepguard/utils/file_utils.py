"""
deepguard/utils/file_utils.py
──────────────────────────────
File validation, media-type detection, and format helpers.
"""

from __future__ import annotations

import mimetypes
import shutil
from enum import Enum, auto
from pathlib import Path


class MediaType(Enum):
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()
    UNKNOWN = auto()


# ── Supported extensions ────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".3gp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a", ".opus", ".wma"}

# Max file sizes (bytes)
MAX_IMAGE_SIZE = 10 * 1024 * 1024   # 10 MB
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500 MB
MAX_AUDIO_SIZE = 50 * 1024 * 1024   # 50 MB


class FileValidationError(ValueError):
    """Raised when a file fails validation."""


def detect_media_type(path: Path) -> MediaType:
    """
    Detect the media type of a file based on its extension and MIME type.
    Extension takes precedence; MIME is used as a fallback.
    """
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return MediaType.IMAGE
    if ext in VIDEO_EXTENSIONS:
        return MediaType.VIDEO
    if ext in AUDIO_EXTENSIONS:
        return MediaType.AUDIO

    # Fallback: MIME type sniffing
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        if mime.startswith("image/"):
            return MediaType.IMAGE
        if mime.startswith("video/"):
            return MediaType.VIDEO
        if mime.startswith("audio/"):
            return MediaType.AUDIO

    return MediaType.UNKNOWN


def validate_file(path: Path, media_type: MediaType | None = None) -> MediaType:
    """
    Validate that a file exists, is readable, has a valid extension,
    and is within the size limit.

    Returns the detected MediaType.
    Raises FileValidationError on any failure.
    """
    if not path.exists():
        raise FileValidationError(f"File not found: {path}")
    if not path.is_file():
        raise FileValidationError(f"Not a file: {path}")

    detected = detect_media_type(path)

    if media_type is not None and detected != media_type:
        raise FileValidationError(
            f"Expected a {media_type.name.lower()} file, but {path.name!r} "
            f"appears to be a {detected.name.lower()} file."
        )

    if detected == MediaType.UNKNOWN:
        raise FileValidationError(
            f"Unsupported file type: {path.suffix!r}\n"
            f"  Images : {', '.join(sorted(IMAGE_EXTENSIONS))}\n"
            f"  Videos : {', '.join(sorted(VIDEO_EXTENSIONS))}\n"
            f"  Audio  : {', '.join(sorted(AUDIO_EXTENSIONS))}"
        )

    size = path.stat().st_size
    limits = {
        MediaType.IMAGE: MAX_IMAGE_SIZE,
        MediaType.VIDEO: MAX_VIDEO_SIZE,
        MediaType.AUDIO: MAX_AUDIO_SIZE,
    }
    max_size = limits[detected]
    if size > max_size:
        raise FileValidationError(
            f"File too large: {_human_size(size)} "
            f"(limit for {detected.name.lower()}: {_human_size(max_size)})"
        )

    return detected


def check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} TB"
