#!/usr/bin/env bash

# Source this file after opening a new shell to restore CUDA/NVCC for this project.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Avoid invalid zero-thread settings that can upset libgomp/MKL.
unset OMP_NUM_THREADS
unset MKL_NUM_THREADS
