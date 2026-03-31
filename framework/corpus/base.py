"""
Base class for all domain corpus fetchers.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from loguru import logger
from rich.console import Console

console = Console()


@dataclass
class CorpusResult:
    texts: List[str] = field(default_factory=list)
    total_chars: int = 0
    total_tokens_approx: int = 0
    sources: List[str] = field(default_factory=list)

    def append(self, text: str, source: str = "") -> None:
        cleaned = text.strip()
        if len(cleaned) < 50:
            return
        self.texts.append(cleaned)
        self.total_chars += len(cleaned)
        self.total_tokens_approx += len(cleaned) // 4  # rough token estimate
        if source and source not in self.sources:
            self.sources.append(source)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sources": self.sources,
                    "total_chars": self.total_chars,
                    "total_tokens_approx": self.total_tokens_approx,
                    "texts": self.texts,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info(f"Corpus saved → {path}  ({len(self.texts)} documents)")

    @classmethod
    def load(cls, path: str | Path) -> "CorpusResult":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cr = cls()
        cr.texts = data["texts"]
        cr.total_chars = data["total_chars"]
        cr.total_tokens_approx = data["total_tokens_approx"]
        cr.sources = data["sources"]
        return cr

    def __len__(self) -> int:
        return len(self.texts)


class BaseCorpus(ABC):
    """Abstract base for all domain corpus fetchers."""

    domain_name: str = "base"
    default_token_budget: int = 10_000_000  # 10 M tokens

    def __init__(self, cache_dir: str = "data/raw", token_budget: int = 0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.token_budget = token_budget or self.default_token_budget
        self._cache_path = self.cache_dir / f"{self.domain_name}_corpus.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, force_refresh: bool = False) -> CorpusResult:
        """Return corpus from cache if available, otherwise fetch."""
        if self._cache_path.exists() and not force_refresh:
            logger.info(f"Loading cached corpus from {self._cache_path}")
            return CorpusResult.load(self._cache_path)

        logger.info(f"Fetching {self.domain_name} corpus (budget: {self.token_budget:,} tokens)…")
        result = CorpusResult()
        self._fetch(result)
        result.save(self._cache_path)
        return result

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _fetch(self, result: CorpusResult) -> None:
        """Populate *result* with domain texts up to self.token_budget."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _budget_remaining(self, result: CorpusResult) -> bool:
        return result.total_tokens_approx < self.token_budget

    def _fetch_hf_dataset(
        self,
        result: CorpusResult,
        dataset_name: str,
        config_name: Optional[str] = None,
        split: str = "train",
        text_field: str = "text",
        max_samples: int = 100_000,
    ) -> None:
        """Convenience: pull a HuggingFace dataset into result."""
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_name, config_name, split=split, streaming=True, trust_remote_code=True)
            count = 0
            for row in ds:
                if not self._budget_remaining(result):
                    break
                txt = row.get(text_field) or row.get("content") or row.get("body") or ""
                if txt:
                    result.append(str(txt), source=f"{dataset_name}/{config_name or split}")
                    count += 1
                if count >= max_samples:
                    break
            logger.info(f"  Loaded {count} samples from {dataset_name}")
        except Exception as e:
            logger.warning(f"  Could not load {dataset_name}: {e}")

    def _fetch_wikipedia(
        self,
        result: CorpusResult,
        lang: str = "en",
        max_articles: int = 5_000,
    ) -> None:
        """Pull random Wikipedia articles via the datasets library."""
        try:
            from datasets import load_dataset
            ds = load_dataset("wikipedia", f"20220301.{lang}", split="train", streaming=True, trust_remote_code=True)
            count = 0
            for row in ds:
                if not self._budget_remaining(result) or count >= max_articles:
                    break
                result.append(row.get("text", ""), source=f"wikipedia-{lang}")
                count += 1
            logger.info(f"  Loaded {count} Wikipedia-{lang} articles")
        except Exception as e:
            logger.warning(f"  Wikipedia fetch failed: {e}")
