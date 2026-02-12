#!/usr/bin/env python3
"""
High-throughput async stress test client for SGLang / vLLM
Addresses: bounded producer-consumer, streaming SSE with TTFT/ITL,
per-request latency percentiles, warmup phase, variable payload shaping.
"""

import asyncio
import json
import time
import random
import sys
from pathlib import Path
from typing import AsyncIterator, Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

import httpx
import aiofiles
import orjson
import numpy as np
from aiolimiter import AsyncLimiter
from rich.console import Console
from rich.live import Live
from rich.table import Table
import typer

console = Console()
app = typer.Typer()

SENTINEL = None  # poison pill for queue shutdown


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class RequestTrace:
    """Per-request timing record"""
    id: str
    latency_s: float
    ttft_s: Optional[float] = None          # time-to-first-token (streaming only)
    itl_ms: Optional[List[float]] = None    # inter-token latencies in ms
    input_tokens: int = 0
    output_tokens: int = 0
    status: str = "success"
    is_warmup: bool = False


@dataclass
class Stats:
    """Aggregated real-time statistics"""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    warmup_completed: int = 0
    warmup_failed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    start_time: float = 0.0
    traces: List[RequestTrace] = field(default_factory=list)

    # ---- derived metrics ----

    @property
    def elapsed(self) -> float:
        return max(time.time() - self.start_time, 1e-9)

    @property
    def tokens_per_second(self) -> float:
        return (self.total_input_tokens + self.total_output_tokens) / self.elapsed

    @property
    def output_tokens_per_second(self) -> float:
        return self.total_output_tokens / self.elapsed

    @property
    def requests_per_second(self) -> float:
        return self.completed_requests / self.elapsed

    def latency_percentiles(self) -> Dict[str, float]:
        """Compute p50 / p95 / p99 from completed, non-warmup traces."""
        lats = [t.latency_s for t in self.traces if t.status == "success"]
        if not lats:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        arr = np.array(lats)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    def ttft_percentiles(self) -> Dict[str, float]:
        vals = [t.ttft_s for t in self.traces if t.ttft_s is not None and t.status == "success"]
        if not vals:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        arr = np.array(vals)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    def itl_percentiles(self) -> Dict[str, float]:
        vals = []
        for t in self.traces:
            if t.itl_ms and t.status == "success":
                vals.extend(t.itl_ms)
        if not vals:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        arr = np.array(vals)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }


# ---------------------------------------------------------------------------
# JSONL reader (streaming, low-memory)
# ---------------------------------------------------------------------------

async def read_jsonl(file_path: Path) -> AsyncIterator[Dict[str, Any]]:
    async with aiofiles.open(file_path, "r") as f:
        async for line in f:
            stripped = line.strip()
            if stripped:
                yield orjson.loads(stripped)


async def count_lines(file_path: Path) -> int:
    count = 0
    async with aiofiles.open(file_path, "r") as f:
        async for line in f:
            if line.strip():
                count += 1
    return count


# ---------------------------------------------------------------------------
# Synthetic payload generator (when no input file is given)
# ---------------------------------------------------------------------------

SYNTHETIC_PROMPTS = [
    "Summarize the key ideas behind transformer architectures in three sentences.",
    "Write a Python function to compute the nth Fibonacci number using memoization.",
    "Explain the difference between supervised and unsupervised learning.",
    "Translate the following English text to French: 'The quick brown fox jumps over the lazy dog.'",
    "What are the main challenges in deploying large language models in production?",
    "Write a short story about a robot discovering emotions for the first time.",
    "List 10 best practices for writing clean, maintainable Python code.",
    "Explain how continuous batching improves LLM serving throughput.",
    "Describe the PagedAttention mechanism used in vLLM.",
    "Write a detailed comparison of SGLang and vLLM inference frameworks.",
]


def generate_synthetic_tasks(
    num_tasks: int,
    prompt_len_distribution: str = "mixed",
) -> List[Dict[str, Any]]:
    """
    Generate synthetic tasks with variable prompt lengths and max_tokens.
    prompt_len_distribution: 'short' | 'long' | 'mixed'
    """
    tasks = []
    for i in range(num_tasks):
        base_prompt = random.choice(SYNTHETIC_PROMPTS)

        if prompt_len_distribution == "short":
            max_tokens = random.randint(32, 128)
        elif prompt_len_distribution == "long":
            # Pad prompt to simulate long prefill (capped to avoid memory explosion)
            padding = " ".join(random.choices(
                ["context", "information", "data", "relevant", "detail"],
                k=random.randint(200, 500),  # capped at 500 words
            ))
            base_prompt = f"Given the following context: {padding}\n\n{base_prompt}"
            max_tokens = random.randint(256, 1024)
        else:  # mixed
            if random.random() < 0.3:
                max_tokens = random.randint(16, 64)
            elif random.random() < 0.7:
                max_tokens = random.randint(64, 256)
            else:
                padding = " ".join(random.choices(
                    ["context", "information", "data", "relevant"],
                    k=random.randint(100, 300),  # capped at 300 words
                ))
                base_prompt = f"Context: {padding}\n\n{base_prompt}"
                max_tokens = random.randint(256, 1024)

        tasks.append({
            "id": str(i),
            "prompt": base_prompt,
            "max_tokens": max_tokens,
            "temperature": random.choice([0.0, 0.0, 0.0, 0.7, 1.0]),  # mostly greedy
        })
    return tasks


# ---------------------------------------------------------------------------
# Request processors (non-streaming & streaming)
# ---------------------------------------------------------------------------

async def process_request_batch(
    client: httpx.AsyncClient,
    api_url: str,
    task: Dict[str, Any],
    max_retries: int = 3,
) -> RequestTrace:
    """Non-streaming request with full latency tracking."""
    payload = {
        "model": task.get("model", "default"),
        "messages": [{"role": "user", "content": task["prompt"]}],
        "temperature": task.get("temperature", 0.0),
        "max_tokens": task.get("max_tokens", 512),
        "stream": False,
    }

    task_id = task.get("id", "unknown")

    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            timeout = httpx.Timeout(10.0, read=180.0)
            resp = await client.post(api_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            latency = time.perf_counter() - t0
            result = resp.json()
            usage = result.get("usage", {})
            return RequestTrace(
                id=task_id,
                latency_s=latency,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                status="success",
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt + random.random())
                continue
            return RequestTrace(id=task_id, latency_s=time.perf_counter() - t0, status="failed")
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt + random.random())
                continue
            return RequestTrace(id=task_id, latency_s=time.perf_counter() - t0, status="failed")

    return RequestTrace(id=task_id, latency_s=0.0, status="failed")


async def process_request_stream(
    client: httpx.AsyncClient,
    api_url: str,
    task: Dict[str, Any],
    max_retries: int = 3,
) -> RequestTrace:
    """Streaming SSE request with TTFT and inter-token latency tracking."""
    payload = {
        "model": task.get("model", "default"),
        "messages": [{"role": "user", "content": task["prompt"]}],
        "temperature": task.get("temperature", 0.0),
        "max_tokens": task.get("max_tokens", 512),
        "stream": True,
    }

    task_id = task.get("id", "unknown")

    for attempt in range(max_retries):
        t0 = time.perf_counter()
        ttft: Optional[float] = None
        itl_timestamps: List[float] = []
        output_text_chunks: List[str] = []
        input_tokens = 0
        output_tokens = 0

        try:
            timeout = httpx.Timeout(10.0, read=300.0)  # longer read timeout for streaming
            async with client.stream("POST", api_url, json=payload, timeout=timeout) as resp:
                resp.raise_for_status()

                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = orjson.loads(data_str)
                    except Exception:
                        continue

                    now = time.perf_counter()

                    # First token
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content and ttft is None:
                        ttft = now - t0

                    if content:
                        output_text_chunks.append(content)
                        itl_timestamps.append(now)

                    # Some servers send usage in the final chunk
                    usage = chunk.get("usage")
                    if usage:
                        input_tokens = usage.get("prompt_tokens", 0)
                        output_tokens = usage.get("completion_tokens", 0)

            latency = time.perf_counter() - t0

            # Compute inter-token latencies
            itl_ms: List[float] = []
            for j in range(1, len(itl_timestamps)):
                itl_ms.append((itl_timestamps[j] - itl_timestamps[j - 1]) * 1000.0)

            # Fallback: approximate tokens from chunks (not accurate - chunks != tokens)
            # If you need accurate counts, ensure server sends usage in final SSE chunk
            if output_tokens == 0:
                output_tokens = len(output_text_chunks)  # rough approximation

            return RequestTrace(
                id=task_id,
                latency_s=latency,
                ttft_s=ttft,
                itl_ms=itl_ms if itl_ms else None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                status="success",
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt + random.random())
                continue
            return RequestTrace(id=task_id, latency_s=time.perf_counter() - t0, status="failed")
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt + random.random())
                continue
            return RequestTrace(id=task_id, latency_s=time.perf_counter() - t0, status="failed")

    return RequestTrace(id=task_id, latency_s=0.0, status="failed")


# ---------------------------------------------------------------------------
# Producer-consumer pipeline (bounded memory)
# ---------------------------------------------------------------------------

async def producer(
    queue: asyncio.Queue,
    tasks: AsyncIterator[Dict[str, Any]] | List[Dict[str, Any]],
    num_consumers: int,
):
    """Feed tasks into the bounded queue. Send sentinel per consumer when done."""
    if isinstance(tasks, list):
        for t in tasks:
            await queue.put(t)
    else:
        async for t in tasks:
            await queue.put(t)

    # Poison pills for each consumer
    for _ in range(num_consumers):
        await queue.put(SENTINEL)


async def consumer(
    queue: asyncio.Queue,
    client: httpx.AsyncClient,
    api_url: str,
    limiter: AsyncLimiter,
    stats: Stats,
    results_queue: asyncio.Queue,
    streaming: bool,
    warmup_ids: set,
    max_retries: int = 3,
):
    """Pull tasks from queue, process, push results."""
    while True:
        task = await queue.get()
        if task is SENTINEL:
            queue.task_done()
            break

        # Acquire rate limit permit before dispatching (don't hold during request)
        await limiter.acquire()

        if streaming:
            trace = await process_request_stream(client, api_url, task, max_retries)
        else:
            trace = await process_request_batch(client, api_url, task, max_retries)

        # Mark warmup traces
        is_warmup = trace.id in warmup_ids
        trace.is_warmup = is_warmup

        # Update stats - warmup tracked separately
        if is_warmup:
            if trace.status == "success":
                stats.warmup_completed += 1
            else:
                stats.warmup_failed += 1
        else:
            stats.traces.append(trace)
            if trace.status == "success":
                stats.completed_requests += 1
                stats.total_input_tokens += trace.input_tokens
                stats.total_output_tokens += trace.output_tokens
            else:
                stats.failed_requests += 1

        await results_queue.put(trace)
        queue.task_done()


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def generate_stats_table(stats: Stats, streaming: bool) -> Table:
    table = Table(title="Stress Test — Live Stats", expand=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    elapsed = stats.elapsed
    warmup_total = stats.warmup_completed + stats.warmup_failed
    measured_total = stats.completed_requests + stats.failed_requests
    table.add_row("Total Requests", f"{stats.total_requests:,}")
    table.add_row("Completed", f"{stats.completed_requests:,}")
    table.add_row("Failed", f"{stats.failed_requests:,}")
    table.add_row("Warmup (excluded)", f"{warmup_total:,} ({stats.warmup_completed} ok, {stats.warmup_failed} fail)")
    pct = (stats.completed_requests / max(measured_total, 1)) * 100
    table.add_row("Success Rate", f"{pct:.1f}%")

    table.add_row("─" * 24, "─" * 28)
    table.add_row("Input Tokens", f"{stats.total_input_tokens:,}")
    table.add_row("Output Tokens", f"{stats.total_output_tokens:,}")
    table.add_row("Throughput (total)", f"{stats.tokens_per_second:,.0f} tok/s")
    table.add_row("Throughput (output)", f"{stats.output_tokens_per_second:,.0f} tok/s")
    table.add_row("Requests/sec", f"{stats.requests_per_second:.1f}")

    table.add_row("─" * 24, "─" * 28)
    lp = stats.latency_percentiles()
    table.add_row("Latency p50", f"{lp['p50']*1000:.1f} ms")
    table.add_row("Latency p95", f"{lp['p95']*1000:.1f} ms")
    table.add_row("Latency p99", f"{lp['p99']*1000:.1f} ms")

    if streaming:
        tp = stats.ttft_percentiles()
        table.add_row("TTFT p50", f"{tp['p50']*1000:.1f} ms")
        table.add_row("TTFT p95", f"{tp['p95']*1000:.1f} ms")
        table.add_row("TTFT p99", f"{tp['p99']*1000:.1f} ms")

        ip = stats.itl_percentiles()
        table.add_row("ITL p50", f"{ip['p50']:.2f} ms")
        table.add_row("ITL p95", f"{ip['p95']:.2f} ms")
        table.add_row("ITL p99", f"{ip['p99']:.2f} ms")

    table.add_row("─" * 24, "─" * 28)
    table.add_row("Elapsed", f"{elapsed:.1f}s")
    if stats.completed_requests > 0 and stats.requests_per_second > 0:
        remaining = stats.total_requests - stats.completed_requests - stats.failed_requests
        eta = remaining / stats.requests_per_second
        table.add_row("ETA", f"{eta:.0f}s ({eta/60:.1f} min)")

    return table


# ---------------------------------------------------------------------------
# Result writer
# ---------------------------------------------------------------------------

async def result_writer(
    results_queue: asyncio.Queue,
    output_file: Optional[Path],
    total: int,
):
    """Drain results queue and write to JSONL."""
    written = 0
    out_f = None
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        out_f = await aiofiles.open(output_file, "w")

    try:
        while written < total:
            trace = await results_queue.get()
            if out_f:
                record = {
                    "id": trace.id,
                    "status": trace.status,
                    "latency_s": round(trace.latency_s, 4),
                    "input_tokens": trace.input_tokens,
                    "output_tokens": trace.output_tokens,
                }
                if trace.ttft_s is not None:
                    record["ttft_s"] = round(trace.ttft_s, 4)
                if trace.itl_ms:
                    record["itl_mean_ms"] = round(np.mean(trace.itl_ms), 2)
                    record["itl_p99_ms"] = round(float(np.percentile(trace.itl_ms, 99)), 2)
                await out_f.write(orjson.dumps(record).decode() + "\n")
            written += 1
    finally:
        if out_f:
            await out_f.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def run(
    api_url: str = typer.Option(
        "http://localhost:8000/v1/chat/completions",
        "--api-url", "-u",
        help="OpenAI-compatible chat completions endpoint",
    ),
    input_file: Optional[Path] = typer.Option(
        None, "--input", "-i",
        help="Input JSONL file. If omitted, synthetic prompts are generated.",
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output JSONL with per-request traces.",
    ),
    num_synthetic: int = typer.Option(
        1000, "--num-tasks", "-n",
        help="Number of synthetic tasks (only when --input is omitted).",
    ),
    prompt_distribution: str = typer.Option(
        "mixed", "--distribution", "-d",
        help="Synthetic prompt length distribution: short | long | mixed",
    ),
    concurrent_requests: int = typer.Option(
        200, "--concurrent", "-c",
        help="Number of concurrent consumer workers.",
    ),
    rate_limit: int = typer.Option(
        500, "--rate-limit", "-r",
        help="Max requests per second.",
    ),
    queue_size: int = typer.Option(
        1000, "--queue-size", "-q",
        help="Bounded task queue size (controls memory).",
    ),
    warmup: int = typer.Option(
        10, "--warmup", "-w",
        help="Number of warmup requests (excluded from percentiles).",
    ),
    streaming: bool = typer.Option(
        False, "--stream", "-s",
        help="Use streaming SSE to measure TTFT and ITL.",
    ),
    max_retries: int = typer.Option(
        3, "--max-retries",
        help="Max retries per request on failure.",
    ),
):
    """
    Stress test an SGLang / vLLM OpenAI-compatible endpoint.

    \b
    Examples:
      # Synthetic mixed workload, streaming, 300 concurrent
      python hyperinfer_stress.py -n 5000 -c 300 -s --rate-limit 1000

      # From JSONL file, non-streaming
      python hyperinfer_stress.py -i prompts.jsonl -o results.jsonl -c 200

      # Long-prefill stress (decode-light)
      python hyperinfer_stress.py -n 2000 -d long -c 100 -s
    """
    asyncio.run(_run_async(
        api_url=api_url,
        input_file=input_file,
        output_file=output_file,
        num_synthetic=num_synthetic,
        prompt_distribution=prompt_distribution,
        concurrent_requests=concurrent_requests,
        rate_limit=rate_limit,
        queue_size=queue_size,
        warmup=warmup,
        streaming=streaming,
        max_retries=max_retries,
    ))


async def _run_async(
    api_url: str,
    input_file: Optional[Path],
    output_file: Optional[Path],
    num_synthetic: int,
    prompt_distribution: str,
    concurrent_requests: int,
    rate_limit: int,
    queue_size: int,
    warmup: int,
    streaming: bool,
    max_retries: int,
):
    stats = Stats(start_time=time.time())

    # ---- Determine tasks ----
    if input_file:
        console.print(f"[cyan]Counting tasks in {input_file}...[/cyan]")
        total = await count_lines(input_file)
        console.print(f"[green]Found {total:,} tasks[/green]")
    else:
        total = num_synthetic
        console.print(f"[cyan]Generating {total:,} synthetic tasks (distribution={prompt_distribution})[/cyan]")

    stats.total_requests = total

    # ---- Warmup task IDs ----
    warmup_ids = {str(i) for i in range(warmup)}

    console.print(f"\n[bold green]Stress Test Config[/bold green]")
    console.print(f"  API:          {api_url}")
    console.print(f"  Streaming:    {streaming}")
    console.print(f"  Concurrency:  {concurrent_requests}")
    console.print(f"  Rate limit:   {rate_limit} req/s")
    console.print(f"  Queue size:   {queue_size}")
    console.print(f"  Warmup:       {warmup} requests")
    console.print(f"  Total tasks:  {total:,}\n")

    # ---- Queues ----
    task_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
    results_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)  # bound to prevent memory growth

    limiter = AsyncLimiter(rate_limit, 1)

    # ---- Build task source ----
    if input_file:
        task_source = read_jsonl(input_file)
    else:
        task_source = generate_synthetic_tasks(total, prompt_distribution)

    stats.start_time = time.time()

    async with httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=concurrent_requests + 50,
            max_keepalive_connections=concurrent_requests,
        ),
    ) as client:

        # Start producer
        prod = asyncio.create_task(
            producer(task_queue, task_source, concurrent_requests)
        )

        # Start consumers
        consumers = [
            asyncio.create_task(consumer(
                task_queue, client, api_url, limiter, stats,
                results_queue, streaming, warmup_ids, max_retries,
            ))
            for _ in range(concurrent_requests)
        ]

        # Start result writer
        writer = asyncio.create_task(
            result_writer(results_queue, output_file, total)
        )

        # Live stats display
        with Live(generate_stats_table(stats, streaming), refresh_per_second=2, console=console) as live:
            while not writer.done():
                await asyncio.sleep(0.5)
                live.update(generate_stats_table(stats, streaming))

        # Await everything
        await prod
        await asyncio.gather(*consumers)
        await writer

    # ---- Final report ----
    console.print("\n" + "═" * 64)
    console.print(generate_stats_table(stats, streaming))
    console.print("═" * 64)

    lp = stats.latency_percentiles()
    console.print(f"\n[bold]Latency  →  p50={lp['p50']*1000:.1f}ms  p95={lp['p95']*1000:.1f}ms  p99={lp['p99']*1000:.1f}ms[/bold]")

    if streaming:
        tp = stats.ttft_percentiles()
        ip = stats.itl_percentiles()
        console.print(f"[bold]TTFT     →  p50={tp['p50']*1000:.1f}ms  p95={tp['p95']*1000:.1f}ms  p99={tp['p99']*1000:.1f}ms[/bold]")
        console.print(f"[bold]ITL      →  p50={ip['p50']:.2f}ms   p95={ip['p95']:.2f}ms   p99={ip['p99']:.2f}ms[/bold]")

    console.print(f"\n[bold green]✓ Done — {stats.completed_requests:,} succeeded, {stats.failed_requests:,} failed[/bold green]")
    if output_file:
        console.print(f"[green]Traces saved to: {output_file}[/green]")
    console.print()


if __name__ == "__main__":
    app()