# =============================================================================
# Optimized Dockerfile for NVIDIA Blackwell (B200 / GB200 NVL72)
# CUDA 12.9.1 | PyTorch 2.8+ | SGLang + sgl-kernel + FlashInfer 0.6.x
# Supports: linux/amd64 (B200), linux/arm64 (Grace Blackwell GB200)
# =============================================================================
#
# Key changes:
#   - CUDA 12.9.1 (12.8.1 does NOT support SM 10.3a; 12.9.1+ required)
#   - sgl-kernel with Blackwell SM100/SM120 kernels
#   - FlashInfer 0.6.x with cu129 wheels (replaces old cu124/torch2.4)
#   - Dropped manual FlashAttention-3 build (SGLang uses CUTLASS MLA / 
#     FlashInfer CuTe DSL attention natively on Blackwell)
#   - TORCH_CUDA_ARCH_LIST targets SM 10.0 (B200) + 10.3 (GB200/B300)
#   - NCCL tuning for NVLink 5th-gen
# =============================================================================

# -- Build args ---------------------------------------------------------------
ARG CUDA_VERSION=12.8.1
ARG PYTHON_VERSION=3.12
ARG SGL_KERNEL_VERSION=0.3.21
ARG FLASHINFER_VERSION=0.6.2
ARG SGLANG_VERSION=0.5.0

# -- Base image ---------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base

ARG TARGETARCH
ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG SGL_KERNEL_VERSION
ARG FLASHINFER_VERSION
ARG SGLANG_VERSION

# -- System packages ----------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python3-pip \
        build-essential \
        ninja-build \
        libibverbs-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && rm -rf /var/lib/apt/lists/*

# -- Install uv (fast Python package manager) ---------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# -- Environment --------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    CUDA_HOME=/usr/local/cuda \
    # ---- Blackwell arch targets ----
    # SM 10.0 = B200 / B100 (sm_100)
    # SM 10.3 = GB200 / GB300 / B300 (sm_103, needs CUDA 12.9+)
    TORCH_CUDA_ARCH_LIST="10.0 10.3" \
    # ---- NCCL tuning for NVLink 5th-gen (GB200 NVL72) ----
    NCCL_MNNVL_ENABLE=1 \
    NCCL_CUMEM_ENABLE=1 \
    NCCL_NET_GDR_LEVEL=5 \
    NCCL_P2P_NVL_CHUNKSIZE=524288 \
    # ---- SGLang Blackwell performance flags ----
    SGL_ENABLE_JIT_DEEPGEMM=1

WORKDIR /app

# -- Virtual environment ------------------------------------------------------
RUN uv venv /opt/venv --python python${PYTHON_VERSION}

# -- PyTorch (CUDA 12.8) -----------------------------------------------------
# cu128 index provides Blackwell-compatible PyTorch wheels
RUN uv pip install --no-cache \
    "torch>=2.5.0" \
    packaging \
    ninja \
    --extra-index-url https://download.pytorch.org/whl/cu128

# -- sgl-kernel (Blackwell SM100/SM120 optimized CUDA kernels) ----------------
# This is CRITICAL for Blackwell — provides RMSNorm, fused ops compiled for SM100+
# Without it you get "no kernel image is available for execution on the device"
RUN uv pip install --no-cache \
    sgl-kernel==${SGL_KERNEL_VERSION}

# -- FlashInfer (cu128 wheels with Blackwell support) -------------------------
RUN uv pip install --no-cache \
    flashinfer-python==${FLASHINFER_VERSION} \
    flashinfer-cubin \
    && uv pip install --no-cache \
    flashinfer-jit-cache==${FLASHINFER_VERSION} \
    --index-url https://flashinfer.ai/whl/cu128

# -- SGLang with Blackwell extras --------------------------------------------
RUN uv pip install --no-cache \
    "sglang[all]>=${SGLANG_VERSION}" \
    hf_transfer \
    requests

# -- (Optional) DeepGEMM for Blackwell MoE/MLA workloads --------------------
# Uncomment for DeepSeek-V3/R1 or other large MoE model serving:
# RUN uv pip install --no-cache \
#     git+https://github.com/sgl-project/DeepGEMM.git@blackwell

# -- Entrypoint ---------------------------------------------------------------
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

# -- Health check -------------------------------------------------------------
# Extended start-period for large model loading on B200 (192GB VRAM)
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]