"""
deepguard/cli.py
──────────────────
Command-line interface for DeepGuard.

Usage:
  deepguard detect <file>               # auto-detect media type
  deepguard detect --image <file>       # force image mode
  deepguard detect --video <file>       # force video mode
  deepguard detect --audio <file>       # force audio mode
  deepguard detect <file> --json        # JSON output
  deepguard detect <file> --output report.json
  deepguard detect <file> --verbose
  deepguard info                        # show config & model registry
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from deepguard import __version__
from deepguard.utils.file_utils import MediaType, detect_media_type, FileValidationError

console = Console()
err_console = Console(stderr=True)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        format="%(levelname)s | %(name)s | %(message)s",
        level=level,
    )


# ── Root group ──────────────────────────────────────────────────────────

@click.group()
@click.version_option(__version__, prog_name="deepguard")
def main():
    """
    \b
    ██████╗ ███████╗███████╗██████╗  ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗
    ██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗
    ██║  ██║█████╗  █████╗  ██████╔╝██║  ███╗██║   ██║███████║██████╔╝██║  ██║
    ██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║
    ██████╔╝███████╗███████╗██║     ╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝
    ╚═════╝ ╚══════╝╚══════╝╚═╝      ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝

    Open-source deepfake detector for images, video, and audio.
    All AI inference is delegated to free external APIs — no GPU required.
    """


# ── detect command ───────────────────────────────────────────────────────

@main.command()
@click.argument("file", type=click.Path(exists=True, readable=True))
@click.option("--image", "force_type", flag_value="image", help="Force image detection mode.")
@click.option("--video", "force_type", flag_value="video", help="Force video detection mode.")
@click.option("--audio", "force_type", flag_value="audio", help="Force audio detection mode.")
@click.option("--json", "output_json", is_flag=True, help="Output result as JSON.")
@click.option(
    "--output", "-o",
    type=click.Path(writable=True),
    default=None,
    help="Save JSON result to a file.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose/debug logging.")
def detect(file: str, force_type: str | None, output_json: bool, output: str | None, verbose: bool):
    """
    Detect whether a FILE is a deepfake / AI-generated media.

    Supports images (.jpg .png .webp …), videos (.mp4 .avi .mkv …),
    and audio (.wav .mp3 .flac .ogg …).
    """
    _setup_logging(verbose)

    # ── Lazy import config here so errors surface cleanly ────────────────
    try:
        from deepguard import config  # noqa: F401 — triggers key validation
    except Exception as exc:
        err_console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        sys.exit(1)

    from deepguard.utils.report import (
        print_image_result, print_video_result, print_audio_result, to_json
    )

    path = Path(file)

    # ── Determine media type ─────────────────────────────────────────────
    if force_type:
        media_type = MediaType[force_type.upper()]
    else:
        media_type = detect_media_type(path)
        if media_type == MediaType.UNKNOWN:
            err_console.print(
                f"[bold red]Error:[/bold red] Cannot detect media type for {path.name!r}.\n"
                "Use --image / --video / --audio to override."
            )
            sys.exit(1)

    console.print(
        Panel.fit(
            f"[bold cyan]DeepGuard v{__version__}[/bold cyan] — "
            f"Analysing [bold]{path.name}[/bold] as [bold]{media_type.name.lower()}[/bold]",
            border_style="cyan",
        )
    )

    # ── Run appropriate detector ─────────────────────────────────────────
    result = None
    try:
        if media_type == MediaType.IMAGE:
            from deepguard.detectors import image_detector
            result = image_detector.detect(path)
            if not output_json:
                print_image_result(result)

        elif media_type == MediaType.VIDEO:
            from deepguard.detectors import video_detector
            result = video_detector.detect(path)
            if not output_json:
                print_video_result(result)

        elif media_type == MediaType.AUDIO:
            from deepguard.detectors import audio_detector
            result = audio_detector.detect(path)
            if not output_json:
                print_audio_result(result)

    except FileValidationError as exc:
        err_console.print(f"[bold red]File error:[/bold red] {exc}")
        sys.exit(1)
    except Exception as exc:
        err_console.print(f"[bold red]Detection failed:[/bold red] {exc}")
        if verbose:
            import traceback
            err_console.print(traceback.format_exc())
        sys.exit(1)

    # ── JSON output ──────────────────────────────────────────────────────
    if result is not None and (output_json or output):
        json_str = to_json(result)
        if output_json and not output:
            console.print(json_str)
        if output:
            out_path = Path(output)
            out_path.write_text(json_str, encoding="utf-8")
            console.print(f"\n[green]✓[/green] JSON report saved to [bold]{out_path}[/bold]")

    # ── Exit code reflects verdict ────────────────────────────────────────
    # Exit 0 = REAL, 1 = FAKE, 2 = UNCERTAIN (useful for scripting)
    if result is not None:
        verdict = getattr(result, "verdict", "UNCERTAIN")
        sys.exit({"REAL": 0, "FAKE": 1, "UNCERTAIN": 2}.get(verdict, 0))


# ── info command ──────────────────────────────────────────────────────────

@main.command()
def info():
    """Show DeepGuard configuration, model registry, and API status."""


    console.print(Panel.fit(
        f"[bold cyan]DeepGuard v{__version__}[/bold cyan] — Configuration",
        border_style="cyan",
    ))

    try:
        from deepguard import config

        # API status
        hf_status = "[green]✓ Set[/green]"
        se_status = (
            "[green]✓ Set (ensemble enabled)[/green]"
            if config.SIGHTENGINE_ENABLED
            else "[yellow]✗ Not set (HuggingFace-only mode)[/yellow]"
        )

        console.print("\n[bold]API Credentials[/bold]")
        console.print(f"  HuggingFace Token : {hf_status}")
        console.print(f"  Sightengine       : {se_status}")
        console.print("\n[bold]Settings[/bold]")
        console.print(f"  Max video frames  : {config.VIDEO_MAX_FRAMES}")
        console.print(f"  Log level         : {config.LOG_LEVEL}")

    except Exception as exc:
        console.print(f"\n  [yellow]⚠ Config error: {exc}[/yellow]")

    # Model registry
    try:
        from deepguard import config
        console.print("\n[bold]Image Detection Models[/bold]")
        for model_id, weight in config.IMAGE_MODELS:
            console.print(f"  [{weight:.0%}] {model_id}")

        console.print("\n[bold]Audio Detection Models[/bold]")
        for model_id, weight in config.AUDIO_MODELS:
            console.print(f"  [{weight:.0%}] {model_id}")

    except Exception:
        pass

    console.print(
        "\n[dim]All models are open-source and hosted on HuggingFace Hub (Apache-2.0 / MIT).[/dim]\n"
    )


if __name__ == "__main__":
    main()
