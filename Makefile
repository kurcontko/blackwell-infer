.PHONY: help install build test clean

help:
	@echo "Blackwell HyperInfer - Available commands:"
	@echo ""
	@echo "  make install       - Install dependencies with uv"
	@echo "  make build         - Build Docker image (local arch)"
	@echo "  make build-multi   - Build multi-arch Docker image"
	@echo "  make test          - Run quick benchmark test"
	@echo "  make calculator    - Run cost calculator"
	@echo "  make clean         - Clean up generated files"
	@echo ""

install:
	@echo "Installing dependencies with uv..."
	uv sync

build:
	@echo "Building Docker image for local architecture..."
	docker build -f docker/Dockerfile -t blackwell-infer:latest .

build-multi:
	@echo "Building multi-arch Docker image..."
	docker buildx build --platform linux/amd64,linux/arm64 \
		-f docker/Dockerfile \
		-t blackwell-infer:latest \
		--push .

test:
	@echo "Running quick test..."
	chmod +x scripts/bench_node.sh
	./scripts/bench_node.sh

calculator:
	@echo "Running cost calculator..."
	uv run scripts/cost_calculator.py compare

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete!"
