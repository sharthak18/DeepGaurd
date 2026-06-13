"""
deepguard/apis/sightengine.py
──────────────────────────────
Optional Sightengine API client for image deepfake detection.

Sightengine is a commercial service with a generous free tier:
  - 2,000 operations/month (deepfake check = 5 ops each → ~400 checks/month)
  - No credit card required
  - Sign up at https://sightengine.com/

This module is only active when SE_API_USER and SE_API_SECRET are set in .env.
If credentials are absent, all calls gracefully return None (skipped).
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests

from deepguard import config

logger = logging.getLogger(__name__)

_SE_API_URL = "https://api.sightengine.com/1.0/check.json"
_SE_VIDEO_SUBMIT_URL = "https://api.sightengine.com/1.0/video/check.json"


class SEAPIError(RuntimeError):
    """Raised on Sightengine API errors."""


def _credentials() -> dict:
    return {
        "api_user": config.SE_API_USER,
        "api_secret": config.SE_API_SECRET,
    }


def check_image(image_path: Path) -> dict | None:
    """
    Submit an image to Sightengine for deepfake detection.

    Returns a dict with the raw Sightengine response, or None if
    Sightengine is not configured / an error occurs (non-fatal).

    Example response structure:
    {
        "status": "success",
        "request": {...},
        "faces": {"num_faces": 1},
        "deepfake": {"score": 0.82}   ← higher = more likely fake
    }
    """
    if not config.SIGHTENGINE_ENABLED:
        return None

    logger.debug("sightengine.check_image: %s", image_path)
    try:
        with open(image_path, "rb") as fh:
            resp = requests.post(
                _SE_API_URL,
                data={**_credentials(), "models": "deepfake,genai"},
                files={"media": fh},
                timeout=60,
            )
        resp.raise_for_status()
        data: dict = resp.json()

        if data.get("status") != "success":
            logger.warning("Sightengine returned non-success: %s", data)
            return None

        return data

    except requests.RequestException as exc:
        logger.warning("Sightengine request failed (non-fatal): %s", exc)
        return None


def extract_deepfake_score(se_response: dict | None) -> float | None:
    """
    Extract a normalised [0.0 – 1.0] fake probability from a Sightengine response.
    Returns None if the response is absent or doesn't contain deepfake data.

    Score semantics: 1.0 = definitely fake, 0.0 = definitely real.
    """
    if se_response is None:
        return None

    deepfake = se_response.get("deepfake", {})
    score = deepfake.get("score")
    if score is None:
        # Try ai_generated field as fallback
        genai = se_response.get("type", {})
        score = genai.get("ai_generated")

    if score is None:
        return None

    return float(score)
