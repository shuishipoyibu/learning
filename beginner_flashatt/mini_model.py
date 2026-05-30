import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _C


@dataclass
class MiniLlamaConfig:
    vocab_size: int = 4096
    hidden_size: int = 256
    num_heads: int = 4
    mlp_ratio: int = 4
    num_layers: int = 2

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        return self.hidden_size // self.num_heads


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(rms + self.eps) * self.weight


class MiniAttention(nn.Module):
    def __init__(self, config: MiniLlamaConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))
        outputs = []
        scale = 1.0 / math.sqrt(self.head_dim)
        for position in range(x.shape[1]):
            q_t = q[:, :, position, :].contiguous()
            k_prefix = k[:, :, : position + 1, :].contiguous()
            v_prefix = v[:, :, : position + 1, :].contiguous()
            scores = torch.einsum("bhd,bhsd->bhs", q_t, k_prefix) * scale
            weights = torch.softmax(scores, dim=-1)
            out_t = torch.einsum("bhs,bhsd->bhd", weights, v_prefix)
            outputs.append(out_t)
        out = torch.stack(outputs, dim=2)
        return self.o_proj(self._merge(out))

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))
        outputs = []
        for position in range(x.shape[1]):
            q_t = q[:, :, position, :].contiguous()
            k_prefix = k[:, :, : position + 1, :].contiguous()
            v_prefix = v[:, :, : position + 1, :].contiguous()
            outputs.append(_C.flash_attention(q_t, k_prefix, v_prefix))
        out = torch.stack(outputs, dim=2)
        return self.o_proj(self._merge(out))


class MiniMLP(nn.Module):
    def __init__(self, config: MiniLlamaConfig) -> None:
        super().__init__()
        intermediate_size = config.hidden_size * config.mlp_ratio
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MiniDecoderLayer(nn.Module):
    def __init__(self, config: MiniLlamaConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size)
        self.post_attn_norm = RMSNorm(config.hidden_size)
        self.attn = MiniAttention(config)
        self.mlp = MiniMLP(config)

    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn.forward_reference(self.input_norm(x))
        x = x + self.mlp(self.post_attn_norm(x))
        return x

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn.forward_cuda(self.input_norm(x))
        x = x + self.mlp(self.post_attn_norm(x))
        return x


class MiniLlamaModel(nn.Module):
    def __init__(self, config: MiniLlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MiniDecoderLayer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def _forward_impl(self, input_ids: torch.Tensor, use_cuda_kernel: bool) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer.forward_cuda(x) if use_cuda_kernel else layer.forward_reference(x)
        x = self.norm(x)
        return self.lm_head(x)

    def forward_reference(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(input_ids, use_cuda_kernel=False)

    def forward_cuda(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(input_ids, use_cuda_kernel=True)
