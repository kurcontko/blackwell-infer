#!/usr/bin/env python3
"""
Model download script for Blackwell HyperInfer
Downloads and caches models to Network Volume for fast startup
"""

import os
from pathlib import Path

import typer
from huggingface_hub import snapshot_download
from rich.console import Console

console = Console()
app = typer.Typer()


@app.command()
def download(
    model_id: str = typer.Argument(..., help="HuggingFace model ID (e.g., Qwen/Qwen2.5-235B-Instruct-FP4)"),
    output_dir: Path = typer.Option(
        Path("/workspace/models"), "--output-dir", "-o", help="Directory to save model (should be on Network Volume)"
    ),
    cache_dir: Path = typer.Option(
        Path("/workspace/hf_cache"),
        "--cache-dir",
        "-c",
        help="HuggingFace cache directory (enables deduplication across models)",
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Skip cache, download directly to output_dir (faster but no deduplication)"
    ),
    token: str | None = typer.Option(
        None, "--token", "-t", help="HuggingFace token for gated models (or set HF_TOKEN env var)"
    ),
    revision: str = typer.Option("main", "--revision", "-r", help="Branch, tag, or commit hash"),
    allow_patterns: list[str] | None = typer.Option(
        None, "--allow", "-a", help="Only download files matching these patterns (e.g., '*.safetensors')"
    ),
    ignore_patterns: list[str] = typer.Option(
        ["*.gguf", "*.md", "consolidated.*"], "--ignore", "-i", help="Skip files matching these patterns"
    ),
):
    """
    Download a model from HuggingFace Hub to local storage.

    Optimized for large models (100GB+) with resumable downloads.
    By default uses cache_dir for deduplication across models.
    Use --no-cache for direct download (faster, no deduplication).
    """

    # Enable HF Transfer for faster downloads (3-5x speedup)
    try:
        import hf_transfer  # noqa: F401

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        console.print("[green]✓ hf_transfer enabled (fast downloads)[/green]")
    except ImportError:
        console.print("[yellow]⚠ hf_transfer not installed — falling back to default[/yellow]")
        console.print("[dim]  pip install hf_transfer for 3-5x speedups[/dim]")

    console.print(f"\n[bold green]Downloading model: {model_id}[/bold green]")
    console.print(f"[yellow]Output directory: {output_dir}[/yellow]")
    if not no_cache:
        console.print(f"[yellow]Cache directory: {cache_dir}[/yellow]")
        console.print("[dim]  (Cache enables deduplication across model downloads)[/dim]")
    else:
        console.print("[dim]Cache disabled - direct download only[/dim]")
    console.print(f"[yellow]Revision: {revision}[/yellow]")
    if ignore_patterns:
        console.print(f"[dim]Ignoring: {', '.join(ignore_patterns)}[/dim]")
    if allow_patterns:
        console.print(f"[dim]Only downloading: {', '.join(allow_patterns)}[/dim]")
    console.print()

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    if not no_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Extract model name for local directory
    model_name = model_id.split("/")[-1].lower()
    local_dir = output_dir / model_name

    # Get token from param or env
    hf_token = token or os.environ.get("HF_TOKEN")

    try:
        console.print("[cyan]Downloading... (progress will appear below)[/cyan]\n")

        if no_cache:
            # Direct download - faster, no deduplication
            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_dir),
                cache_dir=str(local_dir),
                revision=revision,
                resume_download=True,
                max_workers=8,
                local_dir_use_symlinks=False,
                token=hf_token,
                allow_patterns=allow_patterns or None,
                ignore_patterns=ignore_patterns,
            )
        else:
            # Use cache for deduplication (network volume persistent storage)
            # Try symlinks first, fall back to copying if symlinks fail
            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_dir),
                cache_dir=str(cache_dir),
                revision=revision,
                resume_download=True,
                max_workers=8,
                local_dir_use_symlinks="auto",  # Auto-detect symlink support
                token=hf_token,
                allow_patterns=allow_patterns or None,
                ignore_patterns=ignore_patterns,
            )

        console.print("\n[bold green]✓ Model downloaded successfully![/bold green]")
        console.print(f"[green]Location: {local_dir}[/green]")
        console.print("\n[yellow]Set this in your container:[/yellow]")
        console.print(f"[bold]MODEL_PATH={local_dir}[/bold]\n")

    except Exception as e:
        console.print(f"\n[bold red]✗ Download failed: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def verify(
    model_path: Path = typer.Argument(..., help="Path to model directory to verify"),
):
    """
    Verify that a model directory is valid and complete.
    """
    console.print(f"\n[bold]Verifying model at: {model_path}[/bold]\n")

    # config.json is always required
    if (model_path / "config.json").exists():
        console.print("[green]✓ config.json[/green]")
    else:
        console.print("[red]✗ config.json (missing)[/red]")
        console.print("\n[bold red]✗ Model verification failed - config.json is required[/bold red]")
        raise typer.Exit(code=1)

    # Check for tokenizer files (different models use different formats)
    tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
    has_tokenizer = any((model_path / f).exists() for f in tokenizer_files)

    if has_tokenizer:
        found = [f for f in tokenizer_files if (model_path / f).exists()]
        console.print(f"[green]✓ Tokenizer files: {', '.join(found)}[/green]")
    else:
        console.print("[yellow]⚠ No tokenizer files found (may be required)[/yellow]")

    # Check for weight files
    weight_patterns = ["*.safetensors", "*.bin"]
    weight_files = []
    for pattern in weight_patterns:
        weight_files.extend(model_path.glob(pattern))

    if weight_files:
        total_size_gb = sum(f.stat().st_size for f in weight_files) / 1e9
        console.print(f"[green]✓ Found {len(weight_files)} weight files ({total_size_gb:.2f} GB)[/green]")
    else:
        console.print("[red]✗ No weight files found[/red]")
        console.print("\n[bold red]✗ Model verification failed - no weight files[/bold red]")
        raise typer.Exit(code=1)

    console.print("\n[bold green]✓ Model is valid and ready to use![/bold green]\n")


if __name__ == "__main__":
    app()
