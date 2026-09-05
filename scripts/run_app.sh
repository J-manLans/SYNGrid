#!/usr/bin/env bash
set -e

# Run SYNGrid with NVIDIA GPU offload to avoid AMD/Mesa–Triton conflicts.
__NV_PRIME_RENDER_OFFLOAD=1 \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
python3 -m syn_grid
