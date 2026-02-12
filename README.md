# Blackwell Infer

High-throughput LLM inference on NVIDIA Blackwell B200 GPUs.

[![Build Multi-Arch](https://github.com/kurcontko/blackwell-infer/actions/workflows/build-multiarch.yml/badge.svg)](https://github.com/kurcontko/blackwell-infer/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Features

- Native FP4 support for B200 GPUs
- Dual backends: vLLM (recommended) or SGLang
- Multi-arch Docker images (x86/ARM64)
- Async client with crash-safe checkpointing
- 10-20x cheaper than API providers at scale

---

## Performance (1B tokens on 2x B200)

- **Throughput:** 12,000 tokens/s
- **Time:** ~23 hours
- **Cost:** ~$575 (vs ~$15,000 on Claude/OpenAI API)

---

## Quick Start

### 1. Download Model

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install -e .

python scripts/download_model.py \
    Qwen/Qwen2.5-235B-Instruct-FP4 \
    --output-dir /workspace/models
```

### 2. Pull Image

**vLLM** (recommended):
```bash
docker pull ghcr.io/kurcontko/blackwell-infer:vllm
```

**SGLang** (max optimization):
```bash
docker pull ghcr.io/kurcontko/blackwell-infer:sglang
```

### 3. Deploy on RunPod

- **GPUs:** 2x B200
- **Image:** `ghcr.io/kurcontko/blackwell-infer:vllm`
- **Volume:** `/workspace`
- **Port:** 8000
- **Env:**
  ```
  MODEL_PATH=/workspace/models/qwen2.5-235b-instruct-fp4
  QUANTIZATION=modelopt_fp4
  TP_SIZE=2
  ```

### 4. Run Inference

```bash
echo '{"id": "1", "prompt": "Hello", "max_tokens": 100}' > tasks.jsonl

uv run client/stress_test.py \
    --api-url http://YOUR_POD_IP:8000/v1/chat/completions \
    --input tasks.jsonl \
    --output results.jsonl
```

---

## Project Structure

```
├── docker/
│   ├── vllm/       # vLLM backend
│   └── sglang/     # SGLang backend
├── client/
│   └── stress_test.py
├── scripts/
│   ├── download_model.py
│   └── cost_calculator.py
└── configs/
    └── b200_fp4.json
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/workspace/models/qwen2.5-235b-instruct-fp4` | Model location |
| `QUANTIZATION` | `modelopt_fp4` | Quantization (modelopt_fp4/fp8/int4) |
| `TP_SIZE` | Auto | Tensor parallelism |
| `MAX_REQUESTS` | `1024` | Concurrent requests |

---

## License

MIT License - see [LICENSE](LICENSE)
