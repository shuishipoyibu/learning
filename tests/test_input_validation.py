from project_env import sanitize_thread_env

sanitize_thread_env()

import pytest
import torch


def cuda_tensors():
    q = torch.randn(1, 1, 32, device="cuda").contiguous()
    k = torch.randn(1, 1, 16, 32, device="cuda").contiguous()
    v = torch.randn(1, 1, 16, 32, device="cuda").contiguous()
    return q, k, v


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rejects_non_contiguous_tensors():
    import beginner_flashatt

    q, k, v = cuda_tensors()
    bigger = torch.randn(1, 1, 16, 64, device="cuda")
    bad_k = bigger[..., ::2]
    assert bad_k.shape == k.shape
    assert not bad_k.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        beginner_flashatt.standard_decode_attention(q, bad_k, v)
    with pytest.raises(ValueError, match="contiguous"):
        beginner_flashatt.flash_attention(q, bad_k, v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rejects_wrong_dtype():
    import beginner_flashatt

    q, k, v = cuda_tensors()
    with pytest.raises(ValueError, match="float32"):
        beginner_flashatt.standard_decode_attention(q.half(), k, v)
    with pytest.raises(ValueError, match="float32"):
        beginner_flashatt.flash_attention(q.half(), k, v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rejects_shape_mismatch():
    import beginner_flashatt

    q, k, v = cuda_tensors()
    bad_v = torch.randn(1, 1, 17, 32, device="cuda").contiguous()
    with pytest.raises(ValueError, match="same shape"):
        beginner_flashatt.standard_decode_attention(q, k, bad_v)
    with pytest.raises(ValueError, match="same shape"):
        beginner_flashatt.flash_attention(q, k, bad_v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rejects_head_dim_above_teaching_limit():
    import beginner_flashatt

    q = torch.randn(1, 1, 257, device="cuda").contiguous()
    k = torch.randn(1, 1, 16, 257, device="cuda").contiguous()
    v = torch.randn(1, 1, 16, 257, device="cuda").contiguous()
    with pytest.raises(RuntimeError, match="head_dim"):
        beginner_flashatt.standard_decode_attention(q, k, v)
    with pytest.raises(RuntimeError, match="head_dim"):
        beginner_flashatt.flash_attention(q, k, v)
