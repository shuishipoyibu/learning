from project_env import sanitize_thread_env

sanitize_thread_env()

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_mini_model_end_to_end_matches_reference():
    import beginner_flashatt

    torch.manual_seed(7)
    config = beginner_flashatt.MiniLlamaConfig(
        vocab_size=1024,
        hidden_size=128,
        num_heads=4,
        mlp_ratio=4,
        num_layers=2,
    )
    model = beginner_flashatt.MiniLlamaModel(config).cuda().float()
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device="cuda")

    with torch.no_grad():
        ref_logits = model.forward_reference(input_ids)
        cuda_logits = model.forward_cuda(input_ids)

    torch.testing.assert_close(cuda_logits, ref_logits, rtol=2e-4, atol=2e-4)


def test_mini_model_output_shape():
    import beginner_flashatt

    config = beginner_flashatt.MiniLlamaConfig(
        vocab_size=2048,
        hidden_size=256,
        num_heads=4,
        mlp_ratio=4,
        num_layers=1,
    )
    model = beginner_flashatt.MiniLlamaModel(config).cuda().float()
    input_ids = torch.randint(0, config.vocab_size, (1, 16), device="cuda")

    with torch.no_grad():
        logits = model.forward_cuda(input_ids)

    assert logits.shape == (1, 16, config.vocab_size)
    assert logits.is_contiguous()
