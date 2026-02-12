# Blackwell Infer

**High-throughput LLM inference on NVIDIA Blackwell B200** - optimized for batch processing at scale.

<div align="center">

[![Build Multi-Arch](https://github.com/kurcontko/blackwell-infer/actions/workflows/build-multiarch.yml/badge.svg)](https://github.com/kurcontko/blackwell-infer/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 🚀 Features

- ✅ **Native FP4 support** for Blackwell B200 GPUs
- ✅ **Dual backends** - Choose between vLLM (fast builds) or SGLang (max optimization)
- ✅ **Multi-architecture** - Supports x86 (HGX) and ARM64 (Grace Blackwell)
- ✅ **Async client** with crash-safe checkpointing
- ✅ **Cost-effective** - 10-20x cheaper than API providers for large workloads

---

## 📊 Performance (1B Tokens)

| GPU | VRAM | Throughput | Time | Cost |
|-----|------|------------|------|------|
| **2x B200 (FP4)** | 360GB | 12,000 tok/s | ~23 hrs | ~$575 |
| 2x H200 (FP8) | 282GB | 7,000 tok/s | ~40 hrs | ~$720 |

> **Cost comparison:** Processing 1B tokens costs **~$575** on B200 vs **~$15,000** on Claude/OpenAI API

---

## 🛠️ Quick Start

### 1. Download Model (on CPU instance)

```bash
# Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install -e .

# Download model to Network Volume
python scripts/download_model.py \
    Qwen/Qwen2.5-235B-Instruct-FP4 \
    --output-dir /workspace/models
```

### 2. Choose Inference Backend

**vLLM** (Recommended - faster builds, production-stable):
```bash
docker pull ghcr.io/kurcontko/blackwell-infer:vllm
```

**SGLang** (Maximum Blackwell optimization):
```bash
docker pull ghcr.io/kurcontko/blackwell-infer:sglang
```

### 3. Deploy on RunPod

Create pod with:
- **GPUs**: 2x B200
- **Image**: `ghcr.io/kurcontko/blackwell-infer:vllm`
- **Volume**: Mount at `/workspace`
- **Port**: 8000
- **Environment**:
  ```
  MODEL_PATH=/workspace/models/qwen-235b-fp4
  QUANTIZATION=fp4
  TP_SIZE=2
  ```

### 4. Run Inference

```bash
# Prepare your tasks (JSONL format)
echo '{"id": "1", "prompt": "Explain quantum computing", "max_tokens": 500}' > tasks.jsonl

# Run async client
uv run client/stress_test.py \
    --api-url http://YOUR_POD_IP:8000/v1/chat/completions \
    --input tasks.jsonl \
    --output results.jsonl \
    --concurrent 200
```

---

## 📁 Project Structure

```
blackwell-infer/
├── docker/
│   ├── vllm/          # vLLM backend (recommended)
│   └── sglang/        # SGLang backend (max optimization)
├── client/
│   └── stress_test.py # Async high-throughput client
├── scripts/
│   ├── download_model.py    # Model downloader
│   └── cost_calculator.py   # Cost estimator
└── configs/
    └── b200_fp4.json        # B200 configuration
```

---

## 💰 Cost Estimator

```bash
# Calculate for 1 billion tokens
python scripts/cost_calculator.py 1000000000 --gpu 2xB200

# Compare all GPU configs
python scripts/cost_calculator.py compare
```

---

## 🎯 Use Cases

- Large-scale data annotation (millions of examples)
- Synthetic data generation for fine-tuning
- Batch inference on research datasets
- Long-context processing (32K+ tokens)
- Cost-sensitive production workloads

---

## ⚙️ Configuration

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/workspace/models/qwen-235b-fp4` | Model location |
| `QUANTIZATION` | `fp4` | Quantization format (fp4, fp8, int4) |
| `TP_SIZE` | Auto-detect | Tensor parallelism size |
| `MAX_REQUESTS` | `1024` | Max concurrent requests |
| `CONTEXT_LENGTH` | `32768` | Maximum context window |

See `configs/b200_fp4.json` for optimized presets.

---

## 🐛 Troubleshooting

**Server won't start:**
```bash
# Check model exists
ls -lh /workspace/models/

# Verify GPU access
nvidia-smi

# Check logs
docker logs blackwell-vllm
```

**Low throughput:**
```bash
# Check GPU utilization
nvidia-smi dmon

# Increase client concurrency
uv run client/stress_test.py --concurrent 500 --rate-limit 200
```

---

## 📚 References

- [vLLM Documentation](https://docs.vllm.ai/)
- [SGLang Documentation](https://github.com/sgl-project/sglang)
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for processing billions of tokens efficiently** 🚀

</div>
