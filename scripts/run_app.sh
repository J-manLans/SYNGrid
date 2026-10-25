#!/usr/bin/env bash
# run with ./scripts/run_app.sh

set -e

# Run SYNGrid with NVIDIA GPU offload to avoid AMD/Mesa–Triton conflicts.
__NV_PRIME_RENDER_OFFLOAD=1 \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
"$(dirname "$0")/../.venv/bin/python" -m syn_grid
