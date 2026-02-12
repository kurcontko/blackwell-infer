.PHONY: help install build-vllm build-sglang build-multi-vllm build-multi-sglang test clean

help:
	@echo "Blackwell HyperInfer - Available commands:"
	@echo ""
	@echo "  make install           - Install dependencies with uv"
	@echo "  make build-vllm        - Build vLLM Docker image (local arch)"
	@echo "  make build-sglang      - Build SGLang Docker image (local arch)"
	@echo "  make build-multi-vllm  - Build multi-arch vLLM Docker image"
	@echo "  make build-multi-sglang- Build multi-arch SGLang Docker image"
	@echo "  make test              - Run quick benchmark test"
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
