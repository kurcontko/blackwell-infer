#!/usr/bin/env python3
"""
High-throughput async client for Blackwell HyperInfer
Optimized for processing billions of tokens with minimal overhead
"""

import asyncio
import json
import time
from pathlib import Path
from typing import AsyncIterator, Dict, Any
from dataclasses import dataclass, asdict

import httpx
import aiofiles
import orjson
from aiolimiter import AsyncLimiter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.live import Live
from rich.table import Table
import typer

console = Console()
app = typer.Typer()


@dataclass
class Stats:
    """Real-time statistics tracker"""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    start_time: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return (self.total_input_tokens + self.total_output_tokens) / elapsed

    @property
    def requests_per_second(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return self.completed_requests / elapsed


async def read_jsonl(file_path: Path) -> AsyncIterator[Dict[str, Any]]:
    """Stream JSONL file line by line to avoid loading all into memory"""
    async with aiofiles.open(file_path, 'r') as f:
        async for line in f:
            if line.strip():
                yield orjson.loads(line)


async def process_request(
    client: httpx.AsyncClient,
    api_url: str,
    task: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    limiter: AsyncLimiter,
    stats: Stats,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Process a single request with retry logic and backoff"""

    async with semaphore:
        async with limiter:
            payload = {
                "model": "default",
                "messages": [{"role": "user", "content": task["prompt"]}],
                "temperature": task.get("temperature", 0.0),
                "max_tokens": task.get("max_tokens", 512),
            }

            for attempt in range(max_retries):
                try:
                    response = await client.post(
                        api_url,
                        json=payload,
                        timeout=120.0
                    )
                    response.raise_for_status()

                    result = response.json()

                    # Update stats
                    stats.completed_requests += 1
                    if "usage" in result:
                        stats.total_input_tokens += result["usage"].get("prompt_tokens", 0)
                        stats.total_output_tokens += result["usage"].get("completion_tokens", 0)

                    return {
                        "id": task.get("id", "unknown"),
                        "response": result["choices"][0]["message"]["content"],
                        "usage": result.get("usage", {}),
                        "status": "success"
                    }

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429 and attempt < max_retries - 1:
                        # Rate limited - exponential backoff
                        await asyncio.sleep(2 ** attempt)
                        continue
                    stats.failed_requests += 1
                    return {
                        "id": task.get("id", "unknown"),
                        "error": f"HTTP {e.response.status_code}: {str(e)}",
                        "status": "failed"
                    }

                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    stats.failed_requests += 1
                    return {
                        "id": task.get("id", "unknown"),
                        "error": str(e),
                        "status": "failed"
                    }

            stats.failed_requests += 1
            return {
                "id": task.get("id", "unknown"),
                "error": "Max retries exceeded",
                "status": "failed"
            }


def generate_stats_table(stats: Stats, elapsed: float) -> Table:
    """Generate real-time statistics table"""
    table = Table(title="Blackwell HyperInfer - Live Stats", expand=True)

    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Total Requests", f"{stats.total_requests:,}")
    table.add_row("Completed", f"{stats.completed_requests:,}")
    table.add_row("Failed", f"{stats.failed_requests:,}")
    table.add_row("Success Rate", f"{(stats.completed_requests / max(stats.total_requests, 1)) * 100:.2f}%")
    table.add_row("─" * 20, "─" * 20)
    table.add_row("Input Tokens", f"{stats.total_input_tokens:,}")
    table.add_row("Output Tokens", f"{stats.total_output_tokens:,}")
    table.add_row("Total Tokens", f"{stats.total_input_tokens + stats.total_output_tokens:,}")
    table.add_row("─" * 20, "─" * 20)
    table.add_row("Throughput", f"{stats.tokens_per_second:,.2f} tokens/s")
    table.add_row("Requests/sec", f"{stats.requests_per_second:.2f}")
    table.add_row("Elapsed Time", f"{elapsed:.2f}s")

    if stats.completed_requests > 0:
        eta_seconds = (stats.total_requests - stats.completed_requests) / stats.requests_per_second
        table.add_row("ETA", f"{eta_seconds:.0f}s ({eta_seconds / 60:.1f} min)")

    return table


@app.command()
def run(
    api_url: str = typer.Option(
        "http://localhost:8000/v1/chat/completions",
        "--api-url", "-u",
        help="SGLang API endpoint"
    ),
    input_file: Path = typer.Option(
        ...,
        "--input", "-i",
        help="Input JSONL file with prompts"
    ),
    output_file: Path = typer.Option(
        ...,
        "--output", "-o",
        help="Output JSONL file for results"
    ),
    concurrent_requests: int = typer.Option(
        200,
        "--concurrent", "-c",
        help="Maximum concurrent requests"
    ),
    rate_limit: int = typer.Option(
        100,
        "--rate-limit", "-r",
        help="Requests per second limit"
    ),
    checkpoint_interval: int = typer.Option(
        1000,
        "--checkpoint",
        help="Save checkpoint every N requests"
    ),
):
    """
    Run high-throughput inference on a batch of prompts.

    Input file format (JSONL):
    {"id": "1", "prompt": "What is 2+2?", "max_tokens": 100}
    {"id": "2", "prompt": "Explain quantum physics", "max_tokens": 500}
    """
    asyncio.run(_run_async(
        api_url=api_url,
        input_file=input_file,
        output_file=output_file,
        concurrent_requests=concurrent_requests,
        rate_limit=rate_limit,
        checkpoint_interval=checkpoint_interval,
    ))


async def _run_async(
    api_url: str,
    input_file: Path,
    output_file: Path,
    concurrent_requests: int,
    rate_limit: int,
    checkpoint_interval: int,
):
    """Async implementation of the main run loop"""

    stats = Stats(start_time=time.time())
    semaphore = asyncio.Semaphore(concurrent_requests)
    limiter = AsyncLimiter(rate_limit, 1)

    console.print(f"\n[bold green]Starting Blackwell HyperInfer Client[/bold green]")
    console.print(f"[yellow]API: {api_url}[/yellow]")
    console.print(f"[yellow]Input: {input_file}[/yellow]")
    console.print(f"[yellow]Output: {output_file}[/yellow]")
    console.print(f"[yellow]Concurrency: {concurrent_requests} | Rate Limit: {rate_limit} req/s[/yellow]\n")

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Count total tasks
    console.print("[cyan]Counting tasks...[/cyan]")
    async for _ in read_jsonl(input_file):
        stats.total_requests += 1
    console.print(f"[green]Found {stats.total_requests:,} tasks[/green]\n")

    # Reset stats start time
    stats.start_time = time.time()

    async with httpx.AsyncClient() as client:
        async with aiofiles.open(output_file, 'w') as out_file:

            # Create progress display
            with Live(generate_stats_table(stats, 0), refresh_per_second=2) as live:

                tasks = []
                async for task in read_jsonl(input_file):
                    coro = process_request(
                        client=client,
                        api_url=api_url,
                        task=task,
                        semaphore=semaphore,
                        limiter=limiter,
                        stats=stats,
                    )
                    tasks.append(asyncio.create_task(coro))

                # Process results as they complete
                for i, coro in enumerate(asyncio.as_completed(tasks)):
                    result = await coro

                    # Write result immediately (crash-safe)
                    await out_file.write(orjson.dumps(result).decode() + '\n')
                    await out_file.flush()

                    # Update live stats
                    elapsed = time.time() - stats.start_time
                    live.update(generate_stats_table(stats, elapsed))

                    # Checkpoint
                    if (i + 1) % checkpoint_interval == 0:
                        console.log(f"[green]Checkpoint: {i + 1:,} requests processed[/green]")

    # Final summary
    elapsed = time.time() - stats.start_time
    console.print("\n" + "=" * 60)
    console.print(generate_stats_table(stats, elapsed))
    console.print("=" * 60)
    console.print(f"\n[bold green]✓ Processing complete![/bold green]")
    console.print(f"[green]Results saved to: {output_file}[/green]\n")


if __name__ == "__main__":
    app()
