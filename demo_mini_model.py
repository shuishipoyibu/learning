import argparse
from typing import Iterable

from project_env import sanitize_thread_env

sanitize_thread_env()

import torch

import beginner_flashatt


def parse_token_ids(text: str, vocab_size: int) -> list[int]:
    values = []
    for part in text.replace(",", " ").split():
        try:
            token_id = int(part)
        except ValueError as exc:
            raise ValueError(f"invalid token id {part!r}; please enter integers only") from exc
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(f"token id {token_id} is outside valid range [0, {vocab_size})")
        values.append(token_id)
    if not values:
        raise ValueError("please enter at least one token id")
    return values


def format_token_ids(token_ids: Iterable[int]) -> str:
    return " ".join(str(token_id) for token_id in token_ids)


def show_top_predictions(logits: torch.Tensor, top_k: int) -> None:
    values, indices = torch.topk(logits, k=top_k)
    pairs = [f"{idx.item()}:{value.item():.4f}" for value, idx in zip(values, indices)]
    print("top predictions:", ", ".join(pairs))


def run_inference(model: beginner_flashatt.MiniLlamaModel, input_ids: torch.Tensor, top_k: int) -> None:
    with torch.no_grad():
        ref_logits = model.forward_reference(input_ids)
        cuda_logits = model.forward_cuda(input_ids)

    print("input_ids:", format_token_ids(input_ids[0].tolist()))
    print("input_ids shape:", tuple(input_ids.shape))
    print("logits shape:", tuple(cuda_logits.shape))
    print("max |cuda - reference|:", (cuda_logits - ref_logits).abs().max().item())
    show_top_predictions(cuda_logits[0, -1], top_k)


def generate_tokens(
    model: beginner_flashatt.MiniLlamaModel,
    token_ids: list[int],
    steps: int,
    top_k: int,
    device: str,
) -> list[int]:
    generated = list(token_ids)
    for step in range(steps):
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model.forward_cuda(input_ids)
        next_token = int(torch.argmax(logits[0, -1]).item())
        generated.append(next_token)
        print(f"step {step + 1}: next token = {next_token}")
        show_top_predictions(logits[0, -1], top_k)
    return generated


def run_interactive_loop(
    model: beginner_flashatt.MiniLlamaModel,
    vocab_size: int,
    generate_steps: int,
    top_k: int,
    device: str,
) -> None:
    print(f"Enter token ids in [0, {vocab_size}); use spaces or commas. Empty input exits.")
    while True:
        text = input("token ids> ").strip()
        if not text:
            break
        try:
            token_ids = parse_token_ids(text, vocab_size)
        except ValueError as exc:
            print(f"error: {exc}")
            continue

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        run_inference(model, input_ids, top_k)
        if generate_steps > 0:
            generated = generate_tokens(model, token_ids, generate_steps, top_k, device)
            print("generated ids:", format_token_ids(generated))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny LLaMA-style model with the beginner FlashAttention kernel.")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prompt-ids", type=str, default=None, help="Token ids to run, for example: '1 7 42'.")
    parser.add_argument("--interactive", action="store_true", help="Read token ids from stdin and run inference.")
    parser.add_argument("--generate-steps", type=int, default=0, help="Greedily append this many predicted token ids.")
    parser.add_argument("--top-k", type=int, default=5, help="How many next-token predictions to print.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for demo_mini_model.py")
    if args.top_k <= 0 or args.top_k > args.vocab_size:
        raise ValueError("--top-k must be in [1, vocab-size]")
    if args.generate_steps < 0:
        raise ValueError("--generate-steps must be >= 0")

    config = beginner_flashatt.MiniLlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    model = beginner_flashatt.MiniLlamaModel(config).cuda().float()

    if args.interactive:
        run_interactive_loop(model, config.vocab_size, args.generate_steps, args.top_k, "cuda")
        return

    if args.prompt_ids is not None:
        token_ids = parse_token_ids(args.prompt_ids, config.vocab_size)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
        run_inference(model, input_ids, args.top_k)
        if args.generate_steps > 0:
            generated = generate_tokens(model, token_ids, args.generate_steps, args.top_k, "cuda")
            print("generated ids:", format_token_ids(generated))
        return

    input_ids = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device="cuda")
    run_inference(model, input_ids, args.top_k)


if __name__ == "__main__":
    main()
