"""
Transformer language model — built from scratch in PyTorch.

Architecture highlights:
  • Pre-norm (RMSNorm) transformer blocks
  • Rotary Position Embeddings (RoPE)
  • Grouped Query Attention (GQA) — falls back to MHA when num_kv_heads == num_heads
  • SwiGLU feed-forward network
  • Tied input/output embeddings (optional)
  • Works for both SLM (~125 M) and LLM (~1 B) configs via ModelConfig

This is a decoder-only, causal language model suitable for text generation.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.config import ModelConfig, ModelSize


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no bias, no mean subtraction)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------


def _precompute_freqs(dim: int, max_seq: int, theta: float = 10_000.0) -> torch.Tensor:
    """Pre-compute complex rotary frequencies of shape (max_seq, dim//2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq, device=freqs.device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # (max_seq, dim//2) complex


def _apply_rope(
    q: torch.Tensor,   # (B, heads, T, head_dim)
    k: torch.Tensor,   # (B, kv_heads, T, head_dim)
    freqs: torch.Tensor,  # (T, head_dim//2)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    def rotate(x, freqs):
        # x: (B, H, T, D) → view as complex (B, H, T, D/2)
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rot = x_ * freqs.unsqueeze(0).unsqueeze(0)
        return torch.view_as_real(x_rot).flatten(3).to(x.dtype)

    return rotate(q, freqs), rotate(k, freqs)


# ---------------------------------------------------------------------------
# Grouped Query Attention (GQA)
# ---------------------------------------------------------------------------


class GroupedQueryAttention(nn.Module):
    """
    Multi-head attention with optional key-value head grouping (GQA).
    When num_kv_heads == num_heads it behaves as standard MHA.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.hidden_size % cfg.num_heads == 0
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * self.head_dim, cfg.hidden_size, bias=False)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,          # (B, T, D)
        freqs: torch.Tensor,       # (T, head_dim//2)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = _apply_rope(q, k, freqs)

        # Expand KV heads to match Q heads for GQA
        if self.num_kv_heads != self.num_heads:
            ratio = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(ratio, dim=1)
            v = v.repeat_interleave(ratio, dim=1)

        # Scaled dot-product attention (uses flash-attention kernel when available)
        if hasattr(F, "scaled_dot_product_attention"):
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=(mask is None),
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores = scores + mask
            else:
                causal_mask = torch.full((T, T), float("-inf"), device=x.device).triu(1)
                scores = scores + causal_mask
            attn_out = self.dropout(F.softmax(scores.float(), dim=-1).to(q.dtype)) @ v

        out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """SwiGLU FFN: output = SiLU(W1·x) ⊙ (W3·x) projected through W2."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm → Attention → RMSNorm → FFN."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Full Transformer Language Model
# ---------------------------------------------------------------------------


class TransformerLM(nn.Module):
    """
    Decoder-only causal language model.

    Compatible with HuggingFace's generate() when wrapped via
    HuggingFace GenerationMixin, but can also be used standalone.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.embed_dropout = nn.Dropout(cfg.dropout)

        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Pre-compute RoPE frequencies on init (CPU); moved to device on first forward
        head_dim = cfg.hidden_size // cfg.num_heads
        self.register_buffer(
            "_freqs",
            _precompute_freqs(head_dim, cfg.max_seq_len, cfg.rope_theta),
            persistent=False,
        )

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,            # (B, T)
        labels: Optional[torch.Tensor] = None,  # (B, T) — shifted inside
        attention_mask: Optional[torch.Tensor] = None,
    ) -> "TransformerOutput":
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"
        )

        x = self.embed_dropout(self.embed_tokens(input_ids))
        freqs = self._freqs[:T]

        # Build causal mask if padding mask supplied
        mask = None
        if attention_mask is not None:
            # (B, T) → (B, 1, T, T) causal + padding
            pad_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e9
            causal = torch.full((T, T), float("-inf"), device=x.device).triu(1)
            mask = causal.unsqueeze(0).unsqueeze(0) + pad_mask

        for layer in self.layers:
            x = layer(x, freqs, mask)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if labels is not None:
            # Standard next-token prediction loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return TransformerOutput(loss=loss, logits=logits)

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Simple autoregressive generation with top-p nucleus sampling."""
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            ctx = generated[:, -self.cfg.max_seq_len :]
            out = self.forward(ctx)
            next_logits = out.logits[:, -1, :].float()

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(generated.shape[0]):
                    for tok in generated[b].unique():
                        if next_logits[b, tok] < 0:
                            next_logits[b, tok] *= repetition_penalty
                        else:
                            next_logits[b, tok] /= repetition_penalty

            # Temperature
            next_logits = next_logits / max(temperature, 1e-8)

            # Top-K
            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < topk_vals[:, -1:]] = float("-inf")

            # Top-P (nucleus)
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove] = float("-inf")
            next_logits = torch.zeros_like(next_logits).scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)

    def __repr__(self) -> str:
        total = self.num_parameters()
        trainable = self.num_parameters(trainable_only=True)
        return (
            f"TransformerLM("
            f"layers={self.cfg.num_layers}, "
            f"hidden={self.cfg.hidden_size}, "
            f"heads={self.cfg.num_heads}, "
            f"params={total/1e6:.1f}M / trainable={trainable/1e6:.1f}M)"
        )


# ---------------------------------------------------------------------------
# Output dataclass (mimics HuggingFace CausalLMOutputWithPast)
# ---------------------------------------------------------------------------


class TransformerOutput:
    def __init__(self, loss=None, logits=None):
        self.loss = loss
        self.logits = logits

    def __iter__(self):
        yield self.loss
        yield self.logits


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_model(size: ModelSize, cfg: Optional[ModelConfig] = None) -> TransformerLM:
    """Build a TransformerLM from a ModelSize enum (or explicit config)."""
    if cfg is None:
        cfg = ModelConfig.slm() if size == ModelSize.SLM else ModelConfig.llm()
    model = TransformerLM(cfg)
    return model
