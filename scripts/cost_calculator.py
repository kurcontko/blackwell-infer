#!/usr/bin/env python3
"""
Cost and throughput calculator for Blackwell HyperInfer
Estimates time and cost for processing large token volumes
"""

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer()


# GPU configurations (VRAM in GB, Typical throughput in tokens/s)
GPU_CONFIGS = {
    "2xB200": {
        "vram_gb": 360,
        "throughput_low": 8000,  # Conservative estimate (FP8)
        "throughput_high": 12000,  # Optimistic estimate (FP4 + FlashInfer)
        "runpod_price_per_hour": 10.0,  # Approximate as of 2026
    },
    "2xH200": {
        "vram_gb": 282,
        "throughput_low": 5000,
        "throughput_high": 7000,
        "runpod_price_per_hour": 18.0,
    },
    "4xH100": {
        "vram_gb": 320,
        "throughput_low": 4000,
        "throughput_high": 6000,
        "runpod_price_per_hour": 20.0,
    },
}


@app.command()
def calculate(
    total_tokens: int = typer.Argument(..., help="Total number of tokens to process (e.g., 2000000000 for 2B)"),
    gpu_config: str = typer.Option("2xB200", "--gpu", "-g", help=f"GPU configuration: {', '.join(GPU_CONFIGS.keys())}"),
    price_per_hour: float | None = typer.Option(
        None, "--price", "-p", help="Custom price per hour (overrides default)"
    ),
    optimistic: bool = typer.Option(False, "--optimistic", help="Use optimistic throughput estimates"),
):
    """
    Calculate estimated time and cost for processing tokens.

    Example:
        python cost_calculator.py 2000000000 --gpu 2xB200
    """

    if gpu_config not in GPU_CONFIGS:
        console.print(f"[red]Unknown GPU config: {gpu_config}[/red]")
        console.print(f"[yellow]Available: {', '.join(GPU_CONFIGS.keys())}[/yellow]")
        raise typer.Exit(1)

    config = GPU_CONFIGS[gpu_config]
    throughput = config["throughput_high"] if optimistic else config["throughput_low"]
    price = price_per_hour if price_per_hour is not None else config["runpod_price_per_hour"]

    # Calculations
    seconds_total = total_tokens / throughput
    hours_total = seconds_total / 3600
    days_total = hours_total / 24
    total_cost = hours_total * price

    # Cost per million tokens
    cost_per_million = (price / throughput) * 1_000_000 / 3600

    # Create results table
    table = Table(title=f"Blackwell HyperInfer - Cost Estimate ({gpu_config})", expand=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("GPU Configuration", gpu_config)
    table.add_row("Total VRAM", f"{config['vram_gb']} GB")
    table.add_row("Throughput Mode", "Optimistic" if optimistic else "Conservative")
    table.add_row("─" * 30, "─" * 30)
    table.add_row("Total Tokens", f"{total_tokens:,}")
    table.add_row("Throughput", f"{throughput:,} tokens/s")
    table.add_row("─" * 30, "─" * 30)
    table.add_row("Estimated Time", f"{hours_total:.2f} hours ({days_total:.2f} days)")
    table.add_row("Price per Hour", f"${price:.2f}")
    table.add_row("Total Cost", f"${total_cost:,.2f}")
    table.add_row("Cost per 1M Tokens", f"${cost_per_million:.4f}")

    console.print("\n")
    console.print(table)
    console.print("\n")

    # Comparison with API providers
    openai_cost_per_million = 15.0  # GPT-4 Turbo approximate
    claude_cost_per_million = 15.0  # Claude Sonnet approximate

    savings_vs_openai = (openai_cost_per_million * (total_tokens / 1_000_000)) - total_cost
    savings_vs_claude = (claude_cost_per_million * (total_tokens / 1_000_000)) - total_cost

    if savings_vs_openai > 0:
        console.print(
            f"[green]💰 Savings vs OpenAI API: ${savings_vs_openai:,.2f} ({savings_vs_openai / total_cost * 100:.0f}x cheaper)[/green]"
        )
    if savings_vs_claude > 0:
        console.print(
            f"[green]💰 Savings vs Claude API: ${savings_vs_claude:,.2f} ({savings_vs_claude / total_cost * 100:.0f}x cheaper)[/green]"
        )

    console.print("\n[yellow]Note: Estimates assume continuous 100% GPU utilization.[/yellow]")
    console.print("[yellow]Actual costs may vary based on setup time, failures, and network latency.[/yellow]\n")


@app.command()
def compare():
    """Compare all GPU configurations for 1B tokens"""

    console.print("\n[bold]Comparison for 1,000,000,000 tokens[/bold]\n")

    table = Table(expand=True)
    table.add_column("GPU Config", style="cyan")
    table.add_column("VRAM", style="white")
    table.add_column("Throughput", style="yellow")
    table.add_column("Time", style="magenta")
    table.add_column("Cost", style="green")
    table.add_column("$/1M tokens", style="green")

    total_tokens = 1_000_000_000

    for name, config in GPU_CONFIGS.items():
        throughput = config["throughput_high"]
        hours = (total_tokens / throughput) / 3600
        cost = hours * config["runpod_price_per_hour"]
        cost_per_million = (config["runpod_price_per_hour"] / throughput) * 1_000_000 / 3600

        table.add_row(
            name,
            f"{config['vram_gb']} GB",
            f"{throughput:,} tok/s",
            f"{hours:.1f}h ({hours / 24:.1f}d)",
            f"${cost:,.2f}",
            f"${cost_per_million:.4f}",
        )

    console.print(table)
    console.print("\n[yellow]Note: Using optimistic throughput estimates[/yellow]\n")


if __name__ == "__main__":
    app()
