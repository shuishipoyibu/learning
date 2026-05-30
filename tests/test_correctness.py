import math

from project_env import sanitize_thread_env

sanitize_thread_env()

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def torch_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = torch.einsum("bhd,bhsd->bhs", q, k) / math.sqrt(q.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bhs,bhsd->bhd", weights, v)


@pytest.mark.parametrize(
    "batch_size,num_heads,seq_len,head_dim",
    [
        (1, 1, 1, 8),
        (1, 2, 16, 32),
        (2, 4, 128, 64),
        (2, 3, 257, 96),
        (1, 1, 512, 128),
    ],
)
def test_attention_matches_torch(batch_size, num_heads, seq_len, head_dim):
    import beginner_flashatt

    torch.manual_seed(1234 + seq_len + head_dim)
    q = torch.randn(batch_size, num_heads, head_dim, device="cuda").contiguous()
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda").contiguous()
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda").contiguous()

    expected = torch_reference(q, k, v)
    standard = beginner_flashatt.standard_decode_attention(q, k, v)
    flash = beginner_flashatt.flash_attention(q, k, v)

    torch.testing.assert_close(standard, expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(flash, expected, rtol=1e-4, atol=1e-4)


def test_flash_handles_longer_sequence_than_naive_limit():
    import beginner_flashatt

    torch.manual_seed(2026)
    q = torch.randn(1, 1, 32, device="cuda").contiguous()
    k = torch.randn(1, 1, 4097, 32, device="cuda").contiguous()
    v = torch.randn(1, 1, 4097, 32, device="cuda").contiguous()

    expected = torch_reference(q, k, v)
    standard = beginner_flashatt.standard_decode_attention(q, k, v)
    flash = beginner_flashatt.flash_attention(q, k, v)
    torch.testing.assert_close(standard, expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(flash, expected, rtol=1e-4, atol=1e-4)


def test_outputs_are_contiguous_and_shape_correct():
    import beginner_flashatt

    q = torch.randn(2, 2, 64, device="cuda").contiguous()
    k = torch.randn(2, 2, 32, 64, device="cuda").contiguous()
    v = torch.randn(2, 2, 32, 64, device="cuda").contiguous()

    out = beginner_flashatt.flash_attention(q, k, v)
    assert out.shape == q.shape
    assert out.is_contiguous()
    assert out.dtype == torch.float32
    assert out.device.type == "cuda"
