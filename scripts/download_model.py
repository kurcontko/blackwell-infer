#!/usr/bin/env python3
"""
Model download script for Blackwell HyperInfer
Downloads and caches models to Network Volume for fast startup
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
import typer
from rich.console import Console
from rich.progress import Progress

console = Console()
app = typer.Typer()


@app.command()
def download(
    model_id: str = typer.Argument(
        ...,
        help="HuggingFace model ID (e.g., Qwen/Qwen2.5-235B-Instruct-FP4)"
    ),
    output_dir: Path = typer.Option(
        Path("/workspace/models"),
        "--output-dir", "-o",
        help="Directory to save model (should be on Network Volume)"
    ),
    cache_dir: Path = typer.Option(
        Path("/workspace/hf_cache"),
        "--cache-dir", "-c",
        help="HuggingFace cache directory"
    ),
):
    """
    Download a model from HuggingFace Hub to local storage.

    Optimized for large models (100GB+) with resumable downloads.
    """

    # Enable HF Transfer for faster downloads
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    console.print(f"\n[bold green]Downloading model: {model_id}[/bold green]")
    console.print(f"[yellow]Output directory: {output_dir}[/yellow]")
    console.print(f"[yellow]Cache directory: {cache_dir}[/yellow]\n")

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Extract model name for local directory
    model_name = model_id.split("/")[-1].lower()
    local_dir = output_dir / model_name

    try:
        with Progress() as progress:
            task = progress.add_task(
                f"[cyan]Downloading {model_name}...",
                total=None
            )

            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_dir),
                cache_dir=str(cache_dir),
                resume_download=True,
                max_workers=8,
            )

            progress.update(task, completed=True)

        console.print(f"\n[bold green]✓ Model downloaded successfully![/bold green]")
        console.print(f"[green]Location: {local_dir}[/green]")
        console.print(f"\n[yellow]Set this in your container:[/yellow]")
        console.print(f"[bold]MODEL_PATH={local_dir}[/bold]\n")

    except Exception as e:
        console.print(f"\n[bold red]✗ Download failed: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def verify(
    model_path: Path = typer.Argument(
        ...,
        help="Path to model directory to verify"
    ),
):
    """
    Verify that a model directory is valid and complete.
    """
    console.print(f"\n[bold]Verifying model at: {model_path}[/bold]\n")

    required_files = ["config.json", "tokenizer.json"]
    weight_patterns = ["*.safetensors", "*.bin"]

    found_files = []
    missing_files = []

    for file in required_files:
        if (model_path / file).exists():
            found_files.append(file)
            console.print(f"[green]✓ {file}[/green]")
        else:
            missing_files.append(file)
            console.print(f"[red]✗ {file} (missing)[/red]")

    # Check for weight files
    weight_files = []
    for pattern in weight_patterns:
        weight_files.extend(model_path.glob(pattern))

    if weight_files:
        total_size_gb = sum(f.stat().st_size for f in weight_files) / 1e9
        console.print(f"[green]✓ Found {len(weight_files)} weight files ({total_size_gb:.2f} GB)[/green]")
    else:
        console.print(f"[red]✗ No weight files found[/red]")
        missing_files.append("weight files")

    if missing_files:
        console.print(f"\n[bold red]✗ Model verification failed[/bold red]")
        console.print(f"[yellow]Missing: {', '.join(missing_files)}[/yellow]")
        raise typer.Exit(code=1)
    else:
        console.print(f"\n[bold green]✓ Model is valid and ready to use![/bold green]\n")


if __name__ == "__main__":
    app()
