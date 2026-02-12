#!/bin/bash
set -e

# Color output for logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Blackwell HyperInfer - Starting Server${NC}"
echo -e "${GREEN}========================================${NC}"

# ─── Model verification ──────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/workspace/models/qwen-235b-fp4}"

if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}ERROR: Model not found at $MODEL_PATH${NC}"
    echo -e "${YELLOW}Make sure your Network Volume is mounted at /workspace${NC}"
    echo -e "${YELLOW}Run download_model.py first to cache the model${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Model found at: $MODEL_PATH${NC}"

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
COMPUTE_CAP=$(python3 -c "import torch; cc=torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')" 2>/dev/null || echo "0.0")
MAJOR_CAP=$(echo "$COMPUTE_CAP" | cut -d. -f1)

if [ "$MAJOR_CAP" -ge 10 ]; then
    echo -e "${CYAN}✓ Blackwell architecture detected (sm_${COMPUTE_CAP//./_})${NC}"

    # NVLink 5th-gen tuning (multi-GPU / NVL72)
    if [ "$GPU_COUNT" -gt 1 ]; then
        echo -e "${CYAN}  → Enabling NVLink 5th-gen optimizations${NC}"
        export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-1}"
        export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
        export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-5}"
        export NCCL_P2P_NVL_CHUNKSIZE="${NCCL_P2P_NVL_CHUNKSIZE:-524288}"
    fi

    # DeepGEMM JIT compilation for Blackwell
    export SGL_ENABLE_JIT_DEEPGEMM="${SGL_ENABLE_JIT_DEEPGEMM:-1}"
else
    echo -e "${YELLOW}⚠ Non-Blackwell GPU detected (sm_${COMPUTE_CAP//./_}) — Blackwell opts skipped${NC}"
fi

# ─── Package versions ────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}Runtime versions:${NC}"
echo -e "  PyTorch:     $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'missing')"
echo -e "  CUDA:        $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'missing')"
echo -e "  SGLang:      $(python3 -c 'import sglang; print(sglang.__version__)' 2>/dev/null || echo 'missing')"
echo -e "  sgl-kernel:  $(python3 -c 'import sgl_kernel; print(sgl_kernel.__version__)' 2>/dev/null || echo 'missing')"
echo -e "  FlashInfer:  $(python3 -c 'import flashinfer; print(flashinfer.__version__)' 2>/dev/null || echo 'missing')"
echo ""

# Set tensor parallelism based on GPU count
TP_SIZE="${TP_SIZE:-$GPU_COUNT}"
echo -e "${GREEN}✓ Using Tensor Parallelism size: $TP_SIZE${NC}"

# ─── SGLang configuration ────────────────────────────────────────────────────
# CHANGED: --quantization fp4 → modelopt_fp4 (correct SGLang flag for NVFP4 checkpoints)
QUANTIZATION="${QUANTIZATION:-modelopt_fp4}"

KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e5m2}"
MEM_FRACTION="${MEM_FRACTION:-0.95}"
MAX_REQUESTS="${MAX_REQUESTS:-1024}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"

# CHANGED: --enable-flashinfer is deprecated → use --attention-backend
# flashinfer is already the default on recent SGLang, but we set it explicitly
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"

# Chunked prefill size — 8192 is fine for single-node, increase for NVL72
CHUNKED_PREFILL="${CHUNKED_PREFILL:-8192}"

echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Quantization:       $QUANTIZATION"
echo -e "  Attention Backend:  $ATTENTION_BACKEND"
echo -e "  KV Cache:           $KV_CACHE_DTYPE"
echo -e "  Memory Fraction:    $MEM_FRACTION"
echo -e "  Max Concurrent Req: $MAX_REQUESTS"
echo -e "  Context Length:     $CONTEXT_LENGTH"
echo -e "  Chunked Prefill:    $CHUNKED_PREFILL"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Launching SGLang Server...${NC}"
echo -e "${GREEN}========================================${NC}"

# ─── Launch server ────────────────────────────────────────────────────────────
# CHANGES from original:
#   --enable-flashinfer        → --attention-backend flashinfer
#   --quantization fp4         → --quantization modelopt_fp4
#   --tp                       → --tp-size (both work, but --tp-size is canonical)
#   --schedule-policy lpm      → kept (still valid)
#   Added: --trust-remote-code (needed for many models)
exec python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --tp-size "$TP_SIZE" \
    --quantization "$QUANTIZATION" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --attention-backend "$ATTENTION_BACKEND" \
    --mem-fraction-static "$MEM_FRACTION" \
    --max-running-requests "$MAX_REQUESTS" \
    --context-length "$CONTEXT_LENGTH" \
    --chunked-prefill-size "$CHUNKED_PREFILL" \
    --schedule-policy lpm \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    "${@}"