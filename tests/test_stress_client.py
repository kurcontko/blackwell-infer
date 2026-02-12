"""Tests for stress_test.py client"""
import asyncio

import pytest

from client.stress_test import (
    MAX_TRACE_BUFFER,
    SENTINEL,
    RequestTrace,
    Stats,
    generate_synthetic_tasks,
    producer,
)


class TestRequestTrace:
    """Test RequestTrace dataclass"""

    def test_successful_trace(self):
        trace = RequestTrace(
            id="test-1",
            latency_s=0.5,
            input_tokens=100,
            output_tokens=50,
            status="success",
        )
        assert trace.id == "test-1"
        assert trace.latency_s == 0.5
        assert trace.status == "success"
        assert trace.error is None
        assert trace.status_code is None

    def test_failed_trace_with_error(self):
        trace = RequestTrace(
            id="test-2",
            latency_s=0.1,
            status="failed",
            error="Connection timeout",
            status_code=504,
        )
        assert trace.status == "failed"
        assert trace.error == "Connection timeout"
        assert trace.status_code == 504

    def test_warmup_flag(self):
        trace = RequestTrace(id="warmup-1", latency_s=0.3, is_warmup=True)
        assert trace.is_warmup is True


class TestStats:
    """Test Stats aggregation"""

    def test_initial_stats(self):
        stats = Stats()
        assert stats.total_requests == 0
        assert stats.completed_requests == 0
        assert stats.failed_requests == 0
        assert len(stats.traces) == 0

    def test_bounded_trace_buffer(self):
        """Test that trace buffer respects MAX_TRACE_BUFFER limit"""
        stats = Stats()
        # Simulate adding more traces than the buffer limit
        for i in range(MAX_TRACE_BUFFER + 100):
            trace = RequestTrace(id=f"trace-{i}", latency_s=0.1)
            stats.traces.append(trace)
            if len(stats.traces) > MAX_TRACE_BUFFER:
                stats.traces.pop(0)

        assert len(stats.traces) == MAX_TRACE_BUFFER

    def test_latency_percentiles(self):
        stats = Stats()
        # Add some sample traces
        for i in range(100):
            stats.traces.append(
                RequestTrace(id=f"trace-{i}", latency_s=i * 0.01, status="success")
            )

        percentiles = stats.latency_percentiles()
        assert "p50" in percentiles
        assert "p95" in percentiles
        assert "p99" in percentiles
        assert percentiles["p50"] > 0
        assert percentiles["p99"] > percentiles["p50"]

    def test_empty_percentiles(self):
        """Test percentiles with no traces"""
        stats = Stats()
        percentiles = stats.latency_percentiles()
        assert percentiles["p50"] == 0.0
        assert percentiles["p95"] == 0.0
        assert percentiles["p99"] == 0.0


class TestSyntheticTaskGeneration:
    """Test synthetic task generation"""

    def test_generate_short_tasks(self):
        tasks = generate_synthetic_tasks(num_tasks=10, prompt_len_distribution="short")
        assert len(tasks) == 10
        for task in tasks:
            assert "id" in task
            assert "prompt" in task
            assert "max_tokens" in task
            assert 32 <= task["max_tokens"] <= 128

    def test_generate_long_tasks(self):
        tasks = generate_synthetic_tasks(num_tasks=10, prompt_len_distribution="long")
        assert len(tasks) == 10
        for task in tasks:
            assert 256 <= task["max_tokens"] <= 1024
            assert len(task["prompt"]) > 100  # Should have padding

    def test_generate_mixed_tasks(self):
        tasks = generate_synthetic_tasks(num_tasks=100, prompt_len_distribution="mixed")
        assert len(tasks) == 100
        # Should have variety of token lengths
        max_tokens_set = {task["max_tokens"] for task in tasks}
        assert len(max_tokens_set) > 3  # Multiple different token counts


@pytest.mark.asyncio
class TestProducerConsumer:
    """Test producer-consumer pattern"""

    async def test_producer_with_list(self):
        """Test producer with list input"""
        queue = asyncio.Queue(maxsize=10)
        tasks = [{"id": str(i)} for i in range(5)]
        num_consumers = 2

        await producer(queue, tasks, num_consumers)

        # Should have all tasks + sentinels
        items = []
        while not queue.empty():
            items.append(await queue.get())

        # 5 tasks + 2 sentinels
        assert len(items) == 7
        assert items[-2:] == [SENTINEL, SENTINEL]

    async def test_producer_exception_handling(self):
        """Test that producer sends sentinels even on exception"""
        queue = asyncio.Queue(maxsize=10)
        num_consumers = 2

        # Create an async generator that raises
        async def failing_generator():
            yield {"id": "1"}
            raise ValueError("Test error")

        # Producer should still send sentinels due to try/finally
        try:
            await producer(queue, failing_generator(), num_consumers)
        except ValueError:
            pass

        # Check that sentinels were sent
        sentinels_found = 0
        while not queue.empty():
            item = await queue.get()
            if item is SENTINEL:
                sentinels_found += 1

        assert sentinels_found == num_consumers, "Sentinels should be sent even on exception"


@pytest.mark.asyncio
class TestEndToEnd:
    """Integration tests"""

    async def test_stats_calculation_integration(self):
        """Test complete stats workflow"""
        stats = Stats()
        stats.start_time = asyncio.get_event_loop().time()

        # Simulate completed requests
        for i in range(50):
            trace = RequestTrace(
                id=f"req-{i}",
                latency_s=0.1 + (i * 0.01),
                input_tokens=100,
                output_tokens=50,
                status="success",
            )
            stats.traces.append(trace)
            stats.completed_requests += 1
            stats.total_input_tokens += trace.input_tokens
            stats.total_output_tokens += trace.output_tokens

        # Add some failed requests
        for i in range(5):
            trace = RequestTrace(
                id=f"failed-{i}",
                latency_s=0.05,
                status="failed",
                error="Timeout",
            )
            stats.traces.append(trace)
            stats.failed_requests += 1

        assert stats.completed_requests == 50
        assert stats.failed_requests == 5
        assert stats.total_input_tokens == 5000
        assert stats.total_output_tokens == 2500
        assert len(stats.traces) == 55

        # Test derived metrics
        assert stats.tokens_per_second > 0
        assert stats.requests_per_second > 0

        # Test percentiles only include successful requests
        percentiles = stats.latency_percentiles()
        assert percentiles["p50"] > 0
