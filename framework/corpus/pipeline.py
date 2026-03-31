"""
Data pipeline: tokenisation, train/val/test splits, and DataLoader creation.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from loguru import logger
from tqdm.auto import tqdm

from .base import CorpusResult
from framework.config import TrainingConfig


# ---------------------------------------------------------------------------
# Torch Dataset
# ---------------------------------------------------------------------------


class TextDataset(Dataset):
    """
    Sliding-window token dataset.

    Tokenises a list of texts and creates fixed-length windows of `seq_len`
    tokens with a stride of `stride` tokens.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        seq_len: int = 2048,
        stride: int = 512,
        split_name: str = "train",
    ):
        self.seq_len = seq_len
        self.stride = stride
        self.split_name = split_name

        logger.info(f"Tokenising {len(texts)} documents for [{split_name}] split…")
        all_ids: List[int] = []

        for text in tqdm(texts, desc=f"Tokenising [{split_name}]", unit="doc"):
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_token_id or 2)

        # Build windows
        self.windows: List[torch.Tensor] = []
        i = 0
        while i + seq_len < len(all_ids):
            chunk = all_ids[i : i + seq_len + 1]
            self.windows.append(torch.tensor(chunk, dtype=torch.long))
            i += stride

        logger.info(
            f"  [{split_name}] {len(all_ids):,} tokens → {len(self.windows):,} windows "
            f"(seq_len={seq_len}, stride={stride})"
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        window = self.windows[idx]
        input_ids = window[:-1]
        labels = window[1:].clone()
        return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# DataPipeline
# ---------------------------------------------------------------------------


class DataPipeline:
    """
    Orchestrates:
      1. Loading / fetching the corpus
      2. Splitting into train / val / test
      3. Building TextDataset objects
      4. Returning DataLoaders
    """

    def __init__(self, cfg: TrainingConfig, model_cfg, output_dir: str = "data/processed"):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        corpus: CorpusResult,
        tokenizer,
        seed: int = 42,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Returns (train_loader, val_loader, test_loader).
        """
        self._tokenizer = tokenizer
        texts = corpus.texts.copy()

        # Validate split ratios
        total = self.cfg.train_split + self.cfg.val_split + self.cfg.test_split
        assert abs(total - 1.0) < 1e-6, f"Splits must sum to 1.0, got {total}"

        # Shuffle with fixed seed
        rng = random.Random(seed)
        rng.shuffle(texts)

        n = len(texts)
        n_train = int(n * self.cfg.train_split)
        n_val = int(n * self.cfg.val_split)

        train_texts = texts[:n_train]
        val_texts = texts[n_train : n_train + n_val]
        test_texts = texts[n_train + n_val :]

        logger.info(
            f"Corpus split → train:{len(train_texts)} | val:{len(val_texts)} | test:{len(test_texts)}"
        )

        seq_len = self.model_cfg.max_seq_len

        train_ds = TextDataset(train_texts, tokenizer, seq_len=seq_len, stride=seq_len // 4, split_name="train")
        val_ds = TextDataset(val_texts, tokenizer, seq_len=seq_len, stride=seq_len, split_name="val")
        test_ds = TextDataset(test_texts, tokenizer, seq_len=seq_len, stride=seq_len, split_name="test")

        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.per_device_eval_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.cfg.per_device_eval_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        return train_loader, val_loader, test_loader

    # ------------------------------------------------------------------
    # Tokenizer helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_or_train_tokenizer(
        texts: List[str],
        vocab_size: int = 32_000,
        save_path: str = "data/tokenizer",
    ):
        """
        Load a saved tokenizer or train a BPE tokenizer on the corpus.
        Returns a HuggingFace PreTrainedTokenizerFast.
        """
        save_path = Path(save_path)
        tokenizer_file = save_path / "tokenizer.json"

        if tokenizer_file.exists():
            logger.info(f"Loading existing tokenizer from {save_path}")
            from transformers import PreTrainedTokenizerFast
            return PreTrainedTokenizerFast.from_pretrained(str(save_path))

        logger.info(f"Training BPE tokenizer (vocab_size={vocab_size})…")
        save_path.mkdir(parents=True, exist_ok=True)

        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.processors import TemplateProcessing
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from transformers import PreTrainedTokenizerFast

        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"],
            show_progress=True,
        )

        # Use an iterator to avoid loading everything into memory
        def text_iter():
            for t in texts:
                yield t

        tokenizer.train_from_iterator(text_iter(), trainer=trainer, length=len(texts))

        tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            special_tokens=[("<bos>", tokenizer.token_to_id("<bos>")),
                            ("<eos>", tokenizer.token_to_id("<eos>"))],
        )

        # Wrap as HF fast tokenizer
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            unk_token="<unk>",
            mask_token="<mask>",
        )
        fast_tokenizer.save_pretrained(str(save_path))
        logger.info(f"Tokenizer saved → {save_path}")
        return fast_tokenizer
