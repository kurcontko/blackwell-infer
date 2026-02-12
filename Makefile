.PHONY: help install build-vllm build-sglang build-multi-vllm build-multi-sglang test test-unit test-cov test-fast lint format type-check bench calculator clean

help:
	@echo "Blackwell HyperInfer - Available commands:"
	@echo ""
	@echo "  make install           - Install dependencies with uv"
	@echo "  make build-vllm        - Build vLLM Docker image (local arch)"
	@echo "  make build-sglang      - Build SGLang Docker image (local arch)"
	@echo "  make build-multi-vllm  - Build multi-arch vLLM Docker image"
	@echo "  make build-multi-sglang - Build multi-arch SGLang Docker image"
	@echo ""
	@echo "Testing:"
	@echo "  make test              - Run all unit tests"
	@echo "  make test-unit         - Run unit tests only (no integration)"
	@echo "  make test-cov          - Run tests with coverage report"
	@echo "  make test-fast         - Run tests in parallel (faster)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint              - Run Ruff linter"
	@echo "  make format            - Format code with Ruff"
	@echo "  make type-check        - Run mypy type checking"
	@echo "  make pre-commit        - Install pre-commit hooks"
	@echo ""
	@echo "Other:"
	@echo "  make bench             - Run quick benchmark test"
	@echo "  make calculator        - Run cost calculator"
	@echo "  make clean             - Clean up generated files"
	@echo ""

install:
	@echo "Installing dependencies with uv..."
	uv sync

build-vllm:
	@echo "Building vLLM Docker image for local architecture..."
	docker build -f docker/vllm/Dockerfile -t blackwell-infer:vllm docker/vllm

build-sglang:
	@echo "Building SGLang Docker image for local architecture..."
	docker build -f docker/sglang/Dockerfile -t blackwell-infer:sglang docker/sglang

build-multi-vllm:
	@echo "Building multi-arch vLLM Docker image..."
	docker buildx build --platform linux/amd64,linux/arm64 \
		-f docker/vllm/Dockerfile \
		-t ghcr.io/kurcontko/blackwell-infer:vllm \
		--push docker/vllm

build-multi-sglang:
	@echo "Building multi-arch SGLang Docker image..."
	docker buildx build --platform linux/amd64,linux/arm64 \
		-f docker/sglang/Dockerfile \
		-t ghcr.io/kurcontko/blackwell-infer:sglang \
		--push docker/sglang

test:
	@echo "Running all tests..."
	uv run pytest tests/ -v

test-unit:
	@echo "Running unit tests only..."
	uv run pytest tests/ -v -m "not integration and not slow"

test-cov:
	@echo "Running tests with coverage..."
	uv run pytest tests/ --cov=client --cov=scripts --cov-report=term --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

test-fast:
	@echo "Running tests in parallel..."
	uv run pytest tests/ -v -n auto

lint:
	@echo "Running Ruff linter..."
	uv run ruff check .

format:
	@echo "Formatting code with Ruff..."
	uv run ruff format .
	uv run ruff check --fix .

type-check:
	@echo "Running mypy type checking..."
	uv run mypy client/ scripts/ --ignore-missing-imports

pre-commit:
	@echo "Installing pre-commit hooks..."
	uv run pre-commit install
	@echo "Pre-commit hooks installed. Run 'pre-commit run --all-files' to test."

bench:
	@echo "Running quick benchmark test..."
	chmod +x scripts/bench_node.sh
	./scripts/bench_node.sh

calculator:
	@echo "Running cost calculator..."
	uv run scripts/cost_calculator.py compare

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete!"
