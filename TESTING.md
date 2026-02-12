# Testing Guide

This document describes how to run tests and maintain code quality for Blackwell HyperInfer.

## Quick Start

```bash
# Install dependencies
make install

# Run all tests
make test

# Run tests with coverage
make test-cov

# Run linter
make lint

# Format code
make format
```

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures and configuration
├── test_stress_client.py    # Unit tests for stress test client
├── test_download_model.py   # Tests for model downloader
└── test_integration.py      # Integration tests
```

## Running Tests

### All Tests
```bash
make test
# or
uv run pytest tests/ -v
```

### Unit Tests Only (fast)
```bash
make test-unit
# or
uv run pytest tests/ -m "not integration and not slow"
```

### Tests with Coverage
```bash
make test-cov
# or
uv run pytest tests/ --cov=client --cov=scripts --cov-report=html
```

View coverage report: `open htmlcov/index.html`

### Parallel Execution (faster)
```bash
make test-fast
# or
uv run pytest tests/ -n auto
```

## Code Quality

### Linting
```bash
make lint
# or
uv run ruff check .
```

### Formatting
```bash
make format
# or
uv run ruff format .
uv run ruff check --fix .
```

### Type Checking
```bash
make type-check
# or
uv run mypy client/ scripts/ --ignore-missing-imports
```

## Pre-commit Hooks

Install pre-commit hooks to automatically check code before committing:

```bash
make pre-commit
# or
uv run pre-commit install
```

Run manually on all files:
```bash
uv run pre-commit run --all-files
```

### What Pre-commit Checks

- Trailing whitespace
- End-of-file fixer
- YAML/JSON/TOML syntax
- Merge conflict markers
- Ruff linting and formatting
- Type checking with mypy
- Pytest tests (on push only)

## Writing Tests

### Unit Test Example

```python
import pytest
from client.stress_test import RequestTrace

def test_successful_trace():
    trace = RequestTrace(
        id="test-1",
        latency_s=0.5,
        status="success"
    )
    assert trace.status == "success"
    assert trace.error is None
```

### Async Test Example

```python
import pytest

@pytest.mark.asyncio
async def test_producer():
    queue = asyncio.Queue()
    tasks = [{"id": "1"}]
    await producer(queue, tasks, num_consumers=1)
    # assertions...
```

### Using Fixtures

```python
def test_with_temp_dir(temp_dir):
    """Use temp_dir fixture from conftest.py"""
    output_file = temp_dir / "results.jsonl"
    # test code...
```

## Test Markers

### Available Markers

- `@pytest.mark.slow` - Slow tests (skipped by default in fast runs)
- `@pytest.mark.integration` - Integration tests requiring mocks/external services
- `@pytest.mark.asyncio` - Async tests

### Running Specific Markers

```bash
# Run only integration tests
pytest tests/ -m integration

# Skip slow tests
pytest tests/ -m "not slow"

# Run unit tests only
pytest tests/ -m "not integration and not slow"
```

## CI/CD

Tests run automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

CI checks:
1. **Linting** - Ruff linter and formatter
2. **Tests** - pytest on Python 3.11 and 3.12
3. **Type Checking** - mypy (non-blocking)
4. **Coverage** - pytest-cov with Codecov upload

## Coverage Goals

- **Target**: 80%+ coverage for critical paths
- **Priority areas**:
  - Producer-consumer deadlock prevention
  - Bounded buffer management
  - Error handling in request processors
  - Stats calculation accuracy

## Adding New Tests

When adding new features:

1. **Write tests first** (TDD approach)
2. **Test critical paths** - deadlocks, memory leaks, errors
3. **Use fixtures** for common setup
4. **Mark appropriately** - `@pytest.mark.slow`, `@pytest.mark.integration`
5. **Check coverage** - `make test-cov`
6. **Run pre-commit** - `uv run pre-commit run --all-files`

## Debugging Tests

### Run specific test
```bash
pytest tests/test_stress_client.py::TestRequestTrace::test_successful_trace -v
```

### Show print statements
```bash
pytest tests/ -v -s
```

### Drop into debugger on failure
```bash
pytest tests/ -v --pdb
```

### Show locals on failure
```bash
pytest tests/ -v --showlocals
```

## Performance Testing

For performance/benchmark tests:

```python
@pytest.mark.slow
def test_bounded_buffer_performance():
    """Test that bounded buffer maintains O(1) insertion"""
    # Performance test code...
```

Run benchmark tests separately:
```bash
pytest tests/ -m slow -v
```

## Continuous Improvement

- Review test coverage regularly: `make test-cov`
- Add tests for bug fixes
- Keep tests fast (< 1s per test ideal)
- Use mocks for external dependencies
- Update this guide as testing practices evolve
