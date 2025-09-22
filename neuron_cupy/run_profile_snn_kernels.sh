#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root relative to this script
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Pin profiling runs to GPU 3
export CUDA_VISIBLE_DEVICES=3

python "$ROOT_DIR/neuron_cupy/profile_snn_kernels_ncu.py" \
  --kernels baseline optimized \
  --models vit_base llama7b \
  --time-steps 4 8 16 \
  --precisions fp16 fp32 \
  --cuda-visible-devices "$CUDA_VISIBLE_DEVICES" \
  "$@"
