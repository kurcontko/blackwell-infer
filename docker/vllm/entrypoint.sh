#!/bin/bash
set -e

# Color output for logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Blackwell HyperInfer (vLLM) - Starting ${NC}"
echo -e "${GREEN}========================================${NC}"

# ─── Model verification ──────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/workspace/models/qwen-235b-fp4}"

# Support both local paths and HuggingFace model IDs
if [[ "$MODEL_PATH" == /* ]]; then
    # Local path — verify it exists
    if [ ! -d "$MODEL_PATH" ]; then
        echo -e "${RED}ERROR: Model not found at $MODEL_PATH${NC}"
        echo -e "${YELLOW}Make sure your Network Volume is mounted at /workspace${NC}"
        echo -e "${YELLOW}Run download_model.py first to cache the model${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Local model found at: $MODEL_PATH${NC}"
else
    echo -e "${GREEN}✓ Using HuggingFace model: $MODEL_PATH${NC}"
fi

# ─── GPU verification ────────────────────────────────────────────────────────
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: NVIDIA GPU not detected${NC}"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo -e "${GREEN}✓ Detected $GPU_COUNT GPU(s)${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv

# ─── Blackwell detection & arch-specific tuning ──────────────────────────────
COMPUTE_CAP=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || echo "0.0")
MAJOR_CAP=$(echo "$COMPUTE_CAP" | cut -d. -f1)

if [ "$MAJOR_CAP" -ge 10 ]; then
    echo -e "${CYAN}✓ Blackwell architecture detected (sm_${COMPUTE_CAP//./_})${NC}"

    # FlashInfer MoE optimizations for Blackwell
    # These use Blackwell's native FP4 tensor cores via FlashInfer TRTLLM-Gen kernels
    export VLLM_USE_FLASHINFER_MOE_FP8="${VLLM_USE_FLASHINFER_MOE_FP8:-1}"
    export VLLM_USE_FLASHINFER_MOE_FP4="${VLLM_USE_FLASHINFER_MOE_FP4:-1}"
    export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8="${VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8:-1}"

    # NVLink 5th-gen tuning (multi-GPU / NVL72)
    if [ "$GPU_COUNT" -gt 1 ]; then
        echo -e "${CYAN}  → Enabling NVLink 5th-gen optimizations${NC}"
        export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-1}"
        export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
        export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-5}"
        export NCCL_P2P_NVL_CHUNKSIZE="${NCCL_P2P_NVL_CHUNKSIZE:-524288}"
    fi
else
    echo -e "${YELLOW}⚠ Non-Blackwell GPU detected (sm_${COMPUTE_CAP}) — Blackwell opts skipped${NC}"
fi

# ─── Package versions ────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}Runtime versions:${NC}"
echo -e "  PyTorch:     $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'missing')"
echo -e "  CUDA:        $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'missing')"
echo -e "  vLLM:        $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'missing')"
echo -e "  FlashInfer:  $(python3 -c 'import flashinfer; print(flashinfer.__version__)' 2>/dev/null || echo 'missing')"
echo ""

# Set tensor parallelism based on GPU count
TP_SIZE="${TP_SIZE:-$GPU_COUNT}"
echo -e "${GREEN}✓ Using Tensor Parallelism size: $TP_SIZE${NC}"

# ─── vLLM configuration ──────────────────────────────────────────────────────
# Quantization: modelopt_fp4 for NVFP4 checkpoints (Blackwell native FP4 tensor cores)
#               fp8 for FP8 checkpoints
QUANTIZATION="${QUANTIZATION:-modelopt_fp4}"

KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e5m2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

# API server concurrency — decouple HTTP overhead from inference engine
API_SERVER_COUNT="${API_SERVER_COUNT:-8}"

echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Quantization:         $QUANTIZATION"
echo -e "  KV Cache:             $KV_CACHE_DTYPE"
echo -e "  GPU Memory Util:      $GPU_MEM_UTIL"
echo -e "  Max Model Length:     $MAX_MODEL_LEN"
echo -e "  Max Batched Tokens:   $MAX_NUM_BATCHED_TOKENS"
echo -e "  API Server Count:     $API_SERVER_COUNT"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Launching vLLM Server...${NC}"
echo -e "${GREEN}========================================${NC}"

# ─── Launch server ────────────────────────────────────────────────────────────
# vLLM uses `vllm serve` (not python -m vllm.entrypoints...)
# Key Blackwell-specific flags:
#   --async-scheduling        : overlaps CPU/GPU ops for higher throughput
#   --api-server-count N      : decouples HTTP overhead (recommended for high concurrency)
#   --max-num-batched-tokens   : 8192 recommended, 16384 for slightly higher throughput
#   --kv-cache-dtype fp8_e5m2 : halves KV cache memory on Blackwell
#   --quantization modelopt_fp4 : uses Blackwell native FP4 tensor cores
#   --enable-prefix-caching   : improves throughput for shared-prefix workloads
exec vllm serve "$MODEL_PATH" \
    --tensor-parallel-size "$TP_SIZE" \
    --quantization "$QUANTIZATION" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --async-scheduling \
    --enable-prefix-caching \
    --api-server-count "$API_SERVER_COUNT" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    "${@}"