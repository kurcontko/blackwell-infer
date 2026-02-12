# =============================================================================
# Optimized Dockerfile for vLLM on NVIDIA Blackwell (B200 / GB200 NVL72)
# CUDA 12.9 | PyTorch 2.9+ | vLLM 0.12.x + FlashInfer
# Supports: linux/amd64 (B200), linux/arm64 (Grace Blackwell GB200)
# =============================================================================
#
# Strategy: Extend the official vllm/vllm-openai image rather than building
# from scratch. vLLM compiles many CUDA kernels at build time (30+ min),
# and the official wheels already include SM100/SM120 Blackwell targets.
#
# If you need a full source build (e.g. for GB300 with CUDA 13.0), see the
# alternate build instructions at the bottom of this file.
# =============================================================================

# -- Build args ---------------------------------------------------------------
ARG VLLM_VERSION=v0.12.0
ARG BASE_IMAGE=vllm/vllm-openai:${VLLM_VERSION}

# -- Base image ---------------------------------------------------------------
FROM ${BASE_IMAGE}

# -- System extras (add anything your stack needs) ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# -- Additional Python packages -----------------------------------------------
# hf_transfer: 5-10x faster model downloads from HuggingFace Hub
# requests: for health check scripts
RUN pip install --no-cache-dir \
    hf_transfer \
    requests

# -- Environment ---------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    # ---- Blackwell arch targets ----
    TORCH_CUDA_ARCH_LIST="10.0 10.3" \
    # ---- Blackwell FlashInfer MoE optimizations ----
    # These enable FlashInfer's optimized FP8/FP4 MoE kernels on Blackwell
    # (NVIDIA + vLLM co-developed, critical for MoE throughput)
    VLLM_USE_FLASHINFER_MOE_FP8=1 \
    VLLM_USE_FLASHINFER_MOE_FP4=1 \
    VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 \
    # ---- NCCL tuning for NVLink 5th-gen (GB200 NVL72) ----
    NCCL_MNNVL_ENABLE=1 \
    NCCL_CUMEM_ENABLE=1 \
    NCCL_NET_GDR_LEVEL=5 \
    NCCL_P2P_NVL_CHUNKSIZE=524288

WORKDIR /workspace

# -- Entrypoint ---------------------------------------------------------------
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

# -- Health check -------------------------------------------------------------
# Extended start-period for large model loading on B200 (192GB VRAM)
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]


# =============================================================================
# ALTERNATE: Full source build for GB300/B300 (CUDA 13.0) or custom kernels
# Uncomment and use instead of the FROM above:
#
# ARG CUDA_VERSION=13.0.1
# ARG BUILD_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04
#
# DOCKER_BUILDKIT=1 docker build . \
#   --file docker/Dockerfile \
#   --target vllm-openai \
#   --platform "linux/arm64" \
#   --build-arg max_jobs=64 \
#   --build-arg nvcc_threads=2 \
#   --build-arg torch_cuda_arch_list="10.0 10.3+PTX" \
#   --build-arg RUN_WHEEL_CHECK=false \
#   --tag vllm/vllm-gb200-openai:latest
# =============================================================================