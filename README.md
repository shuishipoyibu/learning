# Beginner FlashAttention CUDA

This is a beginner-friendly FlashAttention project built around two core paths:

- `standard_decode_attention`
- `flash_attention` (split + combine)

Both paths target the decode scenario: given one query vector `q` and the current complete KV cache `k` and `v`, they compute the attention output for the current position. The expected tensor shapes are `q[B,H,D]` and `k/v[B,H,S,D]`. This educational implementation currently requires CUDA tensors, `float32` dtype, contiguous memory layout, and `D <= 256`.

## Core Components

- `standard_decode_attention`: uses an on-chip online softmax procedure. For each token, it computes the attention score and updates the global maximum, the softmax denominator, and the accumulated weighted output online. As a result, it does not need to materialize the full attention score matrix.
- `flash_attention`: splits the token sequence into tiles of 16 tokens. The split kernel computes the local maximum, local softmax denominator, and local weighted output for each tile. The combine kernel then uses these per-tile statistics to recover the global normalization weights and merge all tile outputs into the final result.
- `torch_reference`: a PyTorch reference implementation used in the tests and benchmark to compare correctness and performance against the two CUDA paths.

## Project Overview

- `mini_model.py` defines a simple untrained mini model that uses the CUDA `flash_attention` path. Inference can be run through `demo_mini_model.py`.
- `benchmark.py` provides a benchmark script comparing the performance of the three attention paths described above.
- `test_correctness.py` verifies that the three paths produce consistent outputs.
