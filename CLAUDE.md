# Blackwell Infer - Project Context

## Overview
High-throughput LLM inference system optimized for NVIDIA Blackwell B200 GPUs with native FP4 quantization support. This project prioritizes cost efficiency and performance over API convenience.

## Tech Stack
- **Backends**: vLLM (recommended) or SGLang
- **Containerization**: Docker with multi-arch support (x86_64, ARM64)
- **Language**: Python 3.11+
- **Package Manager**: uv (Astral's fast Python package installer)
- **Target Hardware**: NVIDIA B200 GPUs with FP4 support

## Project Structure
```
├── docker/          # Backend-specific Dockerfiles (vllm/, sglang/)
├── client/          # Async inference client with checkpointing
├── scripts/         # Utilities (model download, cost calculator)
├── configs/         # Hardware/model configurations
└── .github/         # CI/CD for multi-arch builds
```

## Domain-Specific Terms
- **TP (Tensor Parallelism)**: Splitting model across multiple GPUs
- **FP4**: 4-bit floating point quantization (native on B200)
- **Checkpointing**: Crash-safe state saving for long inference runs
- **JSONL tasks**: Newline-delimited JSON format for batch inference

## Development Commands

### Testing locally
```bash
# Build vLLM image
make build-vllm

# Run inference test
uv run client/stress_test.py --api-url http://localhost:8000/v1/chat/completions --input example_tasks.jsonl
```

### Adding new configurations
New GPU/model configs go in `configs/`. Use existing `b200_fp4.json` as template.

## Code Conventions
- **Async by default**: Client code uses asyncio for concurrent requests
- **Environment-driven**: Configuration via env vars, not hardcoded paths
- **Fail-safe**: Client must checkpoint progress (inference runs can be 20+ hours)
- **Multi-arch**: All Docker changes must support both x86_64 and ARM64

## Important Files
- `docker-compose.yml`: Local development setup (see lines 1-30 for env vars)
- `client/stress_test.py`: Main inference client with retry logic
- `scripts/download_model.py`: HuggingFace model downloader with resume support
- `.github/workflows/build-multiarch.yml`: CI/CD pipeline

## Known Pitfalls
- **Do not skip checkpointing**: GPU crashes are common on long runs. Client must save state frequently.
- **FP4 vs FP8**: B200 has native FP4 support, but vLLM may default to FP8. Always verify `QUANTIZATION=fp4` in env.
- **Tensor parallelism**: For models > 100B, TP_SIZE must match GPU count. Auto-detection can fail.
- **Memory pressure**: Monitor GPU memory. If OOM occurs, reduce `MAX_REQUESTS` before reducing batch size.

## Deployment Context
- **Primary platform**: RunPod Serverless with 2x B200 GPUs
- **Persistent storage**: Models stored in `/workspace` volume (survives pod restarts)
- **Cost target**: ~$0.58/M tokens (vs ~$15/M on Claude API)

## Testing Strategy
1. Unit tests are minimal - this is infrastructure code
2. Integration testing via `example_tasks.jsonl` against real GPU pod
3. Performance benchmarks tracked in `scripts/cost_calculator.py`

## When Making Changes
- **Docker images**: Test both backends (vLLM and SGLang) before pushing
- **Client code**: Ensure checkpoint/resume works after modifications
- **Configs**: Validate JSON schema matches expected env vars in docker entrypoints
- **Documentation**: Update README.md performance numbers if backend changes affect throughput

## Do Not Touch
- `.github/workflows/build-multiarch.yml` - CI is fragile, changes require testing on ARM runners
- Model quantization code lives in vLLM/SGLang - we don't modify quantization logic

## Getting Help
- vLLM docs: https://docs.vllm.ai/
- SGLang docs: https://github.com/sgl-project/sglang
- RunPod deployment: https://docs.runpod.io/
