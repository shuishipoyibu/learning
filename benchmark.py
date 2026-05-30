import argparse
import math
import time

from project_env import sanitize_thread_env

sanitize_thread_env()

import torch

import beginner_flashatt


def torch_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = torch.einsum("bhd,bhsd->bhs", q, k) / math.sqrt(q.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bhs,bhsd->bhd", weights, v)


def time_cuda(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0 / repeat


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the beginner FlashAttention CUDA extension.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark.py")

    torch.manual_seed(0)
    q = torch.randn(args.batch_size, args.num_heads, args.head_dim, device="cuda").contiguous()
    k = torch.randn(args.batch_size, args.num_heads, args.seq_len, args.head_dim, device="cuda").contiguous()
    v = torch.randn(args.batch_size, args.num_heads, args.seq_len, args.head_dim, device="cuda").contiguous()

    timings = {
        "torch_reference": time_cuda(lambda: torch_reference(q, k, v), args.warmup, args.repeat),
        "standard_decode": time_cuda(
            lambda: beginner_flashatt.standard_decode_attention(q, k, v), args.warmup, args.repeat
        ),
        "flash_tiled_decode": time_cuda(lambda: beginner_flashatt.flash_attention(q, k, v), args.warmup, args.repeat),
    }

    print(f"shape: B={args.batch_size}, H={args.num_heads}, S={args.seq_len}, D={args.head_dim}")
    for name, ms in timings.items():
        print(f"{name:>16}: {ms:.4f} ms")


if __name__ == "__main__":
    main()
