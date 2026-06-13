"""
deepguard/apis/huggingface.py
──────────────────────────────
HuggingFace Inference API client using the official huggingface_hub library.

Uses InferenceClient (the modern, supported approach as of 2025) instead of
raw HTTP calls, which broke when HF migrated from the old /models endpoint
to their Inference Providers router.

Handles:
  - Image classification  (binary: Real / Fake)
  - Audio classification  (binary: Real / Fake / Bona-fide / Spoof)
  - Cold-start retries    (models may need 20-30 s to warm up)
  - Graceful error handling with descriptive messages
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from PIL import Image as PilImage

from deepguard import config

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_COLD_START_WAIT = 25  # seconds


class HFAPIError(RuntimeError):
    """Raised on unrecoverable HuggingFace API errors."""


def _get_client() -> InferenceClient:
    """Return an InferenceClient authenticated with the user's HF token."""
    return InferenceClient(
        provider="hf-inference",
        api_key=config.HF_TOKEN,
    )


# Models that require a specific fixed input resolution (for future use)
_MODEL_INPUT_SIZE: dict[str, int] = {
    # Add model_id: size here if a model requires fixed-size input
    # e.g. "some/model": 384
}


def _prepare_image(image_path: Path, model_id: str) -> Path:
    """
    Return the image path ready for the given model.

    If the model requires a specific input size, resize the image into a
    temporary JPEG file and return that path. The original file is unchanged.
    Otherwise return the original path as-is.
    """
    required_size = _MODEL_INPUT_SIZE.get(model_id)
    if required_size is None:
        return image_path

    img = PilImage.open(image_path).convert("RGB")
    w, h = img.size
    if (w, h) == (required_size, required_size):
        return image_path  # already correct size, no temp file needed

    img = img.resize((required_size, required_size), PilImage.LANCZOS)

    # Write to a properly closed temp file as PNG (server expects PNG/JPEG content-type)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()  # must close before PIL can write on Linux
    img.save(str(tmp_path), format="PNG")
    logger.debug("Resized %s (%dx%d) → %dx%d for %s", image_path.name, w, h, required_size, required_size, model_id)
    return tmp_path


def classify_image(model_id: str, image_path: Path) -> list[dict]:
    """
    Run image classification on a local image file via HuggingFace InferenceClient.

    Returns a list like:
        [{"label": "Fake", "score": 0.97}, {"label": "Real", "score": 0.03}]
    """
    logger.debug("classify_image: model=%s path=%s", model_id, image_path)
    client = _get_client()
    # Resize to required dimensions for models that need fixed input size.
    # _prepare_image returns either the original path (no resize needed) or
    # a Path to a resized temp file. Pass as Path, not str — InferenceClient
    # sets the correct Content-Type automatically for Path objects.
    send_path = _prepare_image(image_path, model_id)

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            # Pass as string path — InferenceClient reads the file and sets the
            # correct Content-Type (image/png or image/jpeg) from the file extension.
            result = client.image_classification(str(send_path), model=model_id)
            # InferenceClient returns a list of ClassificationOutput objects
            return [{"label": item.label, "score": item.score} for item in result]

        except HfHubHTTPError as exc:
            status = getattr(exc.response, "status_code", None) if hasattr(exc, "response") else None

            if status == 503:
                logger.info(
                    "Model %r loading (attempt %d/%d). Waiting %d s …",
                    model_id, attempt, _MAX_RETRIES, _COLD_START_WAIT,
                )
                time.sleep(_COLD_START_WAIT)
                continue

            if status == 429:
                logger.warning("Rate limit hit for %r. Waiting 60 s …", model_id)
                time.sleep(60)
                continue

            if status == 401:
                raise HFAPIError(
                    "HuggingFace returned 401 Unauthorized. "
                    "Check your HF_TOKEN in .env."
                ) from exc

            raise HFAPIError(
                f"HuggingFace API error (HTTP {status}) for model {model_id!r}: {exc}"
            ) from exc

        except Exception as exc:
            # Model might not support inference API — surface clearly
            raise HFAPIError(
                f"Failed to classify image with {model_id!r}: {exc}"
            ) from exc

    raise HFAPIError(
        f"Model {model_id!r} did not become available after {_MAX_RETRIES} retries."
    )


def classify_audio(model_id: str, audio_path: Path) -> list[dict]:
    """
    Run audio classification on a local WAV file via HuggingFace InferenceClient.

    Returns a list like:
        [{"label": "spoof", "score": 0.91}, {"label": "bonafide", "score": 0.09}]
    """
    logger.debug("classify_audio: model=%s path=%s", model_id, audio_path)
    client = _get_client()

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result = client.audio_classification(str(audio_path), model=model_id)
            return [{"label": item.label, "score": item.score} for item in result]

        except HfHubHTTPError as exc:
            status = getattr(exc.response, "status_code", None) if hasattr(exc, "response") else None

            if status == 503:
                logger.info(
                    "Model %r loading (attempt %d/%d). Waiting %d s …",
                    model_id, attempt, _MAX_RETRIES, _COLD_START_WAIT,
                )
                time.sleep(_COLD_START_WAIT)
                continue

            if status == 429:
                logger.warning("Rate limit hit for %r. Waiting 60 s …", model_id)
                time.sleep(60)
                continue

            raise HFAPIError(
                f"HuggingFace API error (HTTP {status}) for model {model_id!r}: {exc}"
            ) from exc

        except Exception as exc:
            raise HFAPIError(
                f"Failed to classify audio with {model_id!r}: {exc}"
            ) from exc

    raise HFAPIError(
        f"Model {model_id!r} did not become available after {_MAX_RETRIES} retries."
    )


def normalize_label(label: str) -> str:
    """
    Normalise heterogeneous model output labels to 'fake' or 'real'.

    Known label vocabularies across all models:
      dima806:    "Fake" / "Real"
      Organika:   "artificial" / "photo"        ← sdxl-detector specific
      umm-maybe:  "artificial" / "photo"
      Wvolf:      "Fake" / "Real"
      MelodyMachine audio: "fake" / "real"
      ASVspoof-style:      "spoof" / "bonafide"
    """
    label_lower = label.lower().strip()
    fake_keywords = {
        "fake", "deepfake", "spoof", "ai", "generated",
        "synthetic", "manipulated", "artificial",  # sdxl-detector uses "artificial"
    }
    real_keywords = {
        "real", "genuine", "bonafide", "bona-fide", "authentic",
        "human", "natural", "photo",               # sdxl-detector uses "photo"
    }

    for kw in fake_keywords:
        if kw in label_lower:
            return "fake"
    for kw in real_keywords:
        if kw in label_lower:
            return "real"

    logger.warning("Unknown label %r — treating as 'unknown'.", label)
    return "unknown"
