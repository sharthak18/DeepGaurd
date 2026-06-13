"""
deepguard/utils/report.py
──────────────────────────
Rich terminal report formatter and JSON exporter.

Produces beautiful, colour-coded verdicts in the terminal and
machine-readable JSON for programmatic consumers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


# ── Result dataclasses ───────────────────────────────────────────────────

@dataclass
class ModelScore:
    model_id: str
    label: str          # 'fake' | 'real' | 'unknown'
    confidence: float   # [0.0 – 1.0] for that label
    raw: list[dict]     # raw API response


@dataclass
class DetectionResult:
    """Single-file detection result (image or audio)."""
    file_path: str
    media_type: str                  # 'image' | 'audio'
    verdict: str                     # 'FAKE' | 'REAL' | 'UNCERTAIN'
    fake_probability: float          # [0.0 – 1.0]
    confidence: float                # ensemble confidence in the verdict
    model_scores: list[ModelScore]
    sightengine_score: float | None  # None when SE not used
    metadata: dict[str, Any]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class FrameResult:
    frame_index: int
    frame_path: str
    result: DetectionResult


@dataclass
class VideoDetectionResult:
    """Video detection result with per-frame breakdown."""
    file_path: str
    media_type: str   # 'video'
    verdict: str
    fake_probability: float
    confidence: float
    frames_analysed: int
    frame_results: list[FrameResult]
    metadata: dict[str, Any]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ── Terminal output ──────────────────────────────────────────────────────

_VERDICT_STYLES = {
    "FAKE":      ("red",    "🚨"),
    "REAL":      ("green",  "✅"),
    "UNCERTAIN": ("yellow", "⚠️ "),
}


def _confidence_bar(prob: float, width: int = 20) -> str:
    filled = int(prob * width)
    return "█" * filled + "░" * (width - filled)


def print_image_result(result: DetectionResult) -> None:
    colour, icon = _VERDICT_STYLES.get(result.verdict, ("white", "❓"))

    # ── Header panel ────────────────────────────────────────────────────
    title = Text(f" {icon}  DeepGuard Detection Report  {icon} ", style=f"bold {colour}")
    console.print(Panel(title, box=box.DOUBLE_EDGE, style=colour))

    # ── Summary ─────────────────────────────────────────────────────────
    console.print(f"\n  [bold]File:[/bold]    {Path(result.file_path).name}")
    console.print(f"  [bold]Type:[/bold]    {result.media_type.capitalize()}")
    console.print(
        f"  [bold]Verdict:[/bold] [bold {colour}]{result.verdict}[/bold {colour}]"
    )
    console.print(
        f"  [bold]Fake probability:[/bold]  "
        f"[{colour}]{_confidence_bar(result.fake_probability)}[/{colour}] "
        f"[bold {colour}]{result.fake_probability:.1%}[/bold {colour}]"
    )
    console.print(
        f"  [bold]Ensemble confidence:[/bold] {result.confidence:.1%}\n"
    )

    # ── Per-model breakdown ──────────────────────────────────────────────
    table = Table(box=box.SIMPLE_HEAD, header_style="bold blue")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Label", justify="center")
    table.add_column("Confidence", justify="right")
    table.add_column("Bar")

    for ms in result.model_scores:
        label_colour = "red" if ms.label == "fake" else "green" if ms.label == "real" else "yellow"
        table.add_row(
            _short_model_id(ms.model_id),
            f"[{label_colour}]{ms.label.upper()}[/{label_colour}]",
            f"{ms.confidence:.1%}",
            f"[{label_colour}]{_confidence_bar(ms.confidence, 15)}[/{label_colour}]",
        )

    if result.sightengine_score is not None:
        se_colour = "red" if result.sightengine_score > 0.5 else "green"
        table.add_row(
            "Sightengine (ensemble)",
            f"[{se_colour}]{'FAKE' if result.sightengine_score > 0.5 else 'REAL'}[/{se_colour}]",
            f"{result.sightengine_score:.1%}",
            f"[{se_colour}]{_confidence_bar(result.sightengine_score, 15)}[/{se_colour}]",
        )

    console.print(table)
    console.print(f"  [dim]Analysed at: {result.timestamp}[/dim]\n")


def print_video_result(result: VideoDetectionResult) -> None:
    colour, icon = _VERDICT_STYLES.get(result.verdict, ("white", "❓"))

    title = Text(f" {icon}  DeepGuard Video Report  {icon} ", style=f"bold {colour}")
    console.print(Panel(title, box=box.DOUBLE_EDGE, style=colour))

    console.print(f"\n  [bold]File:[/bold]    {Path(result.file_path).name}")
    console.print("  [bold]Type:[/bold]    Video")
    console.print(f"  [bold]Frames analysed:[/bold] {result.frames_analysed}")
    console.print(
        f"  [bold]Verdict:[/bold] [bold {colour}]{result.verdict}[/bold {colour}]"
    )
    console.print(
        f"  [bold]Fake probability:[/bold]  "
        f"[{colour}]{_confidence_bar(result.fake_probability)}[/{colour}] "
        f"[bold {colour}]{result.fake_probability:.1%}[/bold {colour}]"
    )
    console.print(
        f"  [bold]Ensemble confidence:[/bold] {result.confidence:.1%}\n"
    )

    # Per-frame summary table
    table = Table(box=box.SIMPLE_HEAD, header_style="bold blue")
    table.add_column("Frame #", justify="right", style="dim")
    table.add_column("Verdict", justify="center")
    table.add_column("Fake Prob.", justify="right")
    table.add_column("Bar")

    for fr in result.frame_results:
        r = fr.result
        fc, _ = _VERDICT_STYLES.get(r.verdict, ("white", ""))
        table.add_row(
            str(fr.frame_index),
            f"[{fc}]{r.verdict}[/{fc}]",
            f"{r.fake_probability:.1%}",
            f"[{fc}]{_confidence_bar(r.fake_probability, 12)}[/{fc}]",
        )

    console.print(table)
    console.print(f"  [dim]Analysed at: {result.timestamp}[/dim]\n")


def print_audio_result(result: DetectionResult) -> None:
    print_image_result(result)  # same layout works for audio


# ── JSON export ──────────────────────────────────────────────────────────

def to_json(result: DetectionResult | VideoDetectionResult, indent: int = 2) -> str:
    """Serialise a detection result to JSON."""
    return json.dumps(_to_dict(result), indent=indent, default=str)


def _to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    return obj


# ── Helpers ──────────────────────────────────────────────────────────────

def _short_model_id(model_id: str) -> str:
    """Shorten a HuggingFace model ID for display."""
    parts = model_id.split("/")
    if len(parts) == 2:
        return f"{parts[0][:10]}…/{parts[1]}" if len(parts[0]) > 10 else model_id
    return model_id
