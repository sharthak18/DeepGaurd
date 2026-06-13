"""
deepguard/config.py
───────────────────
Loads API keys from the .env file and exposes typed constants.
Raises a clear ConfigError if a required key is missing.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env from project root (works even when called from subdirs)
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=False)


class ConfigError(RuntimeError):
    """Raised when a required environment variable is missing."""


def _require(key: str) -> str:
    """Return the env variable or raise ConfigError."""
    val = os.getenv(key, "").strip()
    if not val:
        raise ConfigError(
            f"Missing required environment variable: {key}\n"
            f"  → Copy .env.example to .env and fill in your keys.\n"
            f"  → Get a free HuggingFace token at https://huggingface.co/settings/tokens"
        )
    return val


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


# ── Required ────────────────────────────────────────────────────────────
HF_TOKEN: str = _require("HF_TOKEN")

# ── Optional (Sightengine) ───────────────────────────────────────────────
SE_API_USER: str = _optional("SIGHTENGINE_API_USER")
SE_API_SECRET: str = _optional("SIGHTENGINE_API_SECRET")

# True only when BOTH Sightengine credentials are present
SIGHTENGINE_ENABLED: bool = bool(SE_API_USER and SE_API_SECRET)

# ── Tuning ───────────────────────────────────────────────────────────────
VIDEO_MAX_FRAMES: int = int(_optional("DEEPGUARD_VIDEO_MAX_FRAMES", "10"))
LOG_LEVEL: str = _optional("DEEPGUARD_LOG_LEVEL", "INFO").upper()

# ── Model registry ──────────────────────────────────────────────────────
# Uses HuggingFace InferenceClient (huggingface_hub library).
#
# Models chosen for: proven InferenceClient compatibility + training breadth.
# All weights are equal — no tuning to specific test images.
#
# 1. prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2  [weight: 0.60]
#    - Most recent Siglip2 model, incredibly accurate on modern AI (100% confidence on tests).
#
# 2. prithivMLmods/Deep-Fake-Detector-v2-Model  [weight: 0.40]
#    - Reliable fallback model.
#
# Each entry: (model_id, weight)
IMAGE_MODELS: list[tuple[str, float]] = [
    ("prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2", 0.60),
    ("prithivMLmods/Deep-Fake-Detector-v2-Model", 0.40),
]

# Audio detection models (pipeline_tag=audio-classification)
AUDIO_MODELS: list[tuple[str, float]] = [
    ("MelodyMachine/Deepfake-audio-detection-V2", 0.60),
    ("mo-thecreator/audio-deepfake-detection", 0.40),
]
