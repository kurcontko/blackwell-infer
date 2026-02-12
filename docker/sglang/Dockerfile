# =============================================================================
# Bleeding-Edge Dockerfile for NVIDIA Blackwell (B200 / GB200 NVL72)
# CUDA 13.0.1 | PyTorch 2.10+ (nightly) | SGLang + sgl-kernel + FlashInfer 0.6.x
# Supports: linux/amd64 (B200), linux/arm64 (Grace Blackwell GB200)
# =============================================================================
#
# Key changes from stable build:
#   - CUDA 13.0.1 (latest CUDA with full Blackwell support)
#   - Python 3.13 (latest stable Python)
#   - PyTorch 2.10+ from nightly cu130 index
#   - sgl-kernel with Blackwell SM100/SM120 kernels
#   - FlashInfer 0.6.x with cu130 wheels
#   - TORCH_CUDA_ARCH_LIST targets SM 10.0 (B200) + 10.3 (GB200/B300)
#   - NCCL tuning for NVLink 5th-gen
# =============================================================================

# -- Build args ---------------------------------------------------------------
ARG CUDA_VERSION=13.0.1
ARG PYTHON_VERSION=3.13
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
    # SM 10.3 = GB200 / GB300 / B300 (sm_103, needs CUDA 13.0+)
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

# -- PyTorch (CUDA 13.0 nightly) ---------------------------------------------
# cu130 nightly index provides latest PyTorch 2.10+ with CUDA 13.0 support
# Using nightly for bleeding-edge features and Blackwell optimizations
RUN uv pip install --no-cache \
    torch \
    packaging \
    ninja \
    --index-url https://download.pytorch.org/whl/nightly/cu130

# -- sgl-kernel (Blackwell SM100/SM120 optimized CUDA kernels) ----------------
# This is CRITICAL for Blackwell — provides RMSNorm, fused ops compiled for SM100+
# Without it you get "no kernel image is available for execution on the device"
RUN uv pip install --no-cache \
    sgl-kernel==${SGL_KERNEL_VERSION}

# -- FlashInfer (cu130 wheels with Blackwell support) -------------------------
# Try cu130 first, fallback to cu129 if not available
RUN uv pip install --no-cache \
    flashinfer-python==${FLASHINFER_VERSION} \
    flashinfer-cubin \
    || uv pip install --no-cache \
    flashinfer-python==${FLASHINFER_VERSION} \
    flashinfer-cubin
RUN uv pip install --no-cache \
    flashinfer-jit-cache==${FLASHINFER_VERSION} \
    --index-url https://flashinfer.ai/whl/cu130 \
    || uv pip install --no-cache \
    flashinfer-jit-cache==${FLASHINFER_VERSION} \
    --index-url https://flashinfer.ai/whl/cu129

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