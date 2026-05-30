import torch

try:
    from . import _C
except ImportError as exc:
    raise ImportError(
        "beginner_flashatt CUDA extension is not built. "
        "Run `python setup.py develop` from the project root first."
    ) from exc

from .mini_model import MiniLlamaConfig, MiniLlamaModel


def _check_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q, k, and v must be CUDA tensors")
    if q.dtype != torch.float32 or k.dtype != torch.float32 or v.dtype != torch.float32:
        raise ValueError("this beginner version supports float32 only")
    if q.ndim != 3 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("expected q [B,H,D], k [B,H,S,D], v [B,H,S,D]")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("q, k, and v must be contiguous")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[2] != k.shape[3]:
        raise ValueError("shape mismatch: q [B,H,D], k/v [B,H,S,D]")


def standard_decode_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Standard decode attention kernel that scans the whole KV cache without KV tiling."""
    _check_inputs(q, k, v)
    return _C.standard_decode_attention(q, k, v)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Split-KV tiled FlashAttention decode kernel."""
    _check_inputs(q, k, v)
    return _C.flash_attention(q, k, v)


__all__ = [
    "MiniLlamaConfig",
    "MiniLlamaModel",
    "flash_attention",
    "standard_decode_attention",
]
