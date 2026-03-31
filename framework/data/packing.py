"""
Greedy sequence packing for language-model pre-training.

Instead of padding short sequences to ``max_seq_len`` (which wastes 40–60 % of
compute on meaningless pad tokens), we concatenate **all** tokenised texts —
separated by EOS — into a single flat token stream and split it into
fixed-length, non-overlapping windows.  This eliminates all padding in the
training set and typically increases effective GPU utilisation by 1.4–2×.

Visual sketch
-------------

  doc1 tokens … <eos>  doc2 tokens … <eos>  doc3 tokens … <eos>  …
  ├─── chunk 0 ───────────┤├─── chunk 1 ───────┤├─── chunk 2 ───────┤

Every chunk is exactly ``seq_len`` tokens wide; the corresponding label
tensor is the same chunk shifted left by one position (next-token prediction).

Usage
-----
    packed_ds = PackedDataset(texts, tokenizer, seq_len=2048)
    loader    = DataLoader(packed_ds, batch_size=8, shuffle=True)
    batch     = next(iter(loader))
    # batch["input_ids"]  → (8, 2048)
    # batch["labels"]     → (8, 2048)  — labels == input_ids for causal LM

Notes
-----
- No padding tokens are ever produced; every position in every chunk carries
  a real token from the corpus.
- Document boundaries inside a chunk are marked only by EOS tokens — the model
  learns to handle cross-document context, which is standard practice.
- The last incomplete chunk is **discarded** (``drop_last=True`` semantics) to
  keep all tensors the same size.
- Tokenisation is done eagerly on ``__init__``; for very large corpora
  (>100 M tokens) consider using the streaming variant in
  ``framework.corpus.pipeline`` instead.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from loguru import logger
from tqdm.auto import tqdm


class PackedDataset(Dataset):
    """
    Padding-free, fixed-length sequence dataset built by greedy packing.

    Parameters
    ----------
    texts:
        List of raw text strings to tokenise and pack.
    tokenizer:
        Any HuggingFace ``PreTrainedTokenizerFast``-compatible tokenizer that
        has ``encode(text)`` and an ``eos_token_id`` attribute.
    seq_len:
        Context window length in tokens.  Every yielded sample will be exactly
        ``seq_len`` input tokens (labels are the same length).
    split_name:
        Human-readable label used only in log messages.
    max_tokens:
        Optional hard cap on the total number of tokens added to the flat
        stream (useful to limit memory usage during unit tests).
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        seq_len: int = 2048,
        split_name: str = "train",
        max_tokens: Optional[int] = None,
    ) -> None:
        self.seq_len = seq_len
        self.split_name = split_name

        eos_id: int = tokenizer.eos_token_id or 2

        logger.info(
            f"[PackedDataset:{split_name}] tokenising {len(texts):,} docs "
            f"(seq_len={seq_len}) …"
        )

        # ── 1. Build the flat token stream ──────────────────────────────────
        flat: List[int] = []
        for text in tqdm(texts, desc=f"Packing [{split_name}]", unit="doc", leave=False):
            ids = tokenizer.encode(text, add_special_tokens=False)
            flat.extend(ids)
            flat.append(eos_id)           # document boundary marker

            if max_tokens is not None and len(flat) >= max_tokens:
                flat = flat[:max_tokens]
                break

        total_tokens = len(flat)

        # ── 2. Discard the trailing partial chunk ────────────────────────────
        # We need seq_len+1 tokens per chunk (input + next-token label).
        chunk_width = seq_len + 1
        n_chunks = total_tokens // chunk_width
        usable   = n_chunks * chunk_width

        if n_chunks == 0:
            raise ValueError(
                f"[PackedDataset:{split_name}] Not enough tokens to form even "
                f"one chunk.  Have {total_tokens} tokens but need at least "
                f"{chunk_width}.  Reduce seq_len or increase corpus size."
            )

        flat_tensor = torch.tensor(flat[:usable], dtype=torch.long)
        # Shape → (n_chunks, seq_len+1)
        self._chunks: torch.Tensor = flat_tensor.view(n_chunks, chunk_width)

        # ── 3. Stats ─────────────────────────────────────────────────────────
        padding_waste_pct = (total_tokens - usable) / max(total_tokens, 1) * 100
        logger.info(
            f"[PackedDataset:{split_name}] "
            f"{total_tokens:,} tokens → {n_chunks:,} chunks × {seq_len} "
            f"| tail discarded: {total_tokens - usable:,} tokens "
            f"({padding_waste_pct:.1f}% of stream, not padding)"
        )

    # ── Dataset protocol ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self._chunks[idx]          # (seq_len + 1,)
        input_ids = chunk[:-1].clone()     # (seq_len,)
        labels    = chunk[1:].clone()      # (seq_len,)  next-token targets
        return {"input_ids": input_ids, "labels": labels}

    # ── Helpers ─────────────────────────────────────────────────────────────

    def num_tokens(self) -> int:
        """Total number of real (non-padding) tokens in the dataset."""
        return len(self._chunks) * self.seq_len

    def __repr__(self) -> str:
        return (
            f"PackedDataset(split={self.split_name!r}, "
            f"chunks={len(self):,}, seq_len={self.seq_len})"
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def make_packed_dataset(
    texts: List[str],
    tokenizer,
    seq_len: int,
    split_name: str = "train",
    max_tokens: Optional[int] = None,
) -> PackedDataset:
    """
    Thin wrapper around :class:`PackedDataset` with keyword argument names that
    mirror ``TextDataset`` so callers can swap the two without changing call
    sites.
    """
    return PackedDataset(
        texts=texts,
        tokenizer=tokenizer,
        seq_len=seq_len,
        split_name=split_name,
        max_tokens=max_tokens,
    )
